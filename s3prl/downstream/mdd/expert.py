# Code for: TIMIT Linear downstream expert for linear phoneme recognition
# Will probably not be super useful but it is a good starting point


# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ Kornel Szabo ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import random
import yaml
from collections import defaultdict, Counter
from typing import Dict, List

# -------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import ctc_decoder
from torchaudio.functional import edit_distance
from pathlib import Path
from einops import rearrange
from speechbrain.decoders import ctc_greedy_decode
from speechbrain.utils.edit_distance import (
    wer_details_for_batch,
    accumulatable_wer_stats,
    wer_details_by_utterance,
)

# from pyctcdecode import build_ctcdecoder

# -------------#
from .model import Model
from .dataset import L2ArcticDataset


# This is just copied from the TIMIT phone recognition task
# I haven't understood what the things do just yet
class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim

        # These get loaded by `run_downstream.py` from `./config.yaml`
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        self.dataset = {}
        self.dataset["train"] = L2ArcticDataset("train", **self.datarc)
        self.dataset["dev"] = L2ArcticDataset("dev", **self.datarc)
        self.dataset["test"] = L2ArcticDataset("test", **self.datarc)

        # The final layer we append to the upstream model for L1 and L2 prediction
        self.l1_model = Model(
            input_dim=self.upstream_dim,
            output_class_num=self.dataset["train"].output_class_num,
            **self.modelrc,
        )

        self.l2_model = Model(
            input_dim=self.upstream_dim,
            output_class_num=self.dataset["train"].output_class_num,
            **self.modelrc,
        )

        with open(Path(self.datarc["phone_path"]) / "tokens.txt", "r") as stream:
            arpa_tuples = [(i, phone) for (i, phone) in enumerate(stream.readlines())]
            self.arpa_phones = {phone.strip(): i for (i, phone) in arpa_tuples}
            self.arpa_phone_list = [phone for (i, phone) in arpa_tuples]

        # Blank is the empty label. In this case, it is the label of silence
        self.objective = nn.CTCLoss(blank=self.arpa_phones["sil"], reduction="none")

        # Create mapping from
        # self.decoder = ctc_decoder(
        #     lexicon=None,
        #     tokens=f"{self.datarc['phone_path']}/tokens.txt",
        #     lm=None,  # no language model used
        #     lm_dict=None,
        #     nbest=1,  # beam search returns top 1 best result
        #     beam_size=50,  # we retain 50 beams during beam search
        #     beam_size_token=None,
        #     beam_threshold=50,
        #     lm_weight=2,
        #     word_score=0,
        #     unk_score=-float("inf"),
        #     sil_score=0,
        #     log_add=False,
        #     blank_token="sil",
        #     sil_token="sil",
        #     unk_word="<unk>",
        # )

        self.logging = Path(expdir) / "log.log"
        self.best = defaultdict(lambda: 0)

    # Interface
    def get_dataloader(self, split):
        """
        Dataloader Specs:
            Each dataloader should output in the following format:

            (
                [wav1, wav2, ...],
                [l1_label1, l1_label2, ...],
                [l2_label1, l2_label2, ...],
            )

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==44100
            and
            l1_label1, ... are LongTensors with the encodings of the target label
            sequence
        """
        assert split in [
            "train",
            "dev",
            "test",
        ], f"Invalid split {split}. Expected 'train', 'dev' or 'test'."
        return DataLoader(
            self.dataset[split],
            batch_size=1,
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=self.dataset[split].collate_fn,
        )

    # Interface
    """runner.py releavant code:
        loss = self.downstream.model(
            train_split,
            features, *others,
            records = records,
        )
    """
    def forward(
        self,
        split: str,
        features: List[torch.FloatTensor],
        l1_labels: torch.FloatTensor,
        l2_labels: torch.FloatTensor,
        l1_lengths: torch.LongTensor,
        l2_lengths: torch.LongTensor,
        records: Dict[List[any]],
        **kwargs,
    ):
        """
        Args:
            split:
                indicates which split we are (e.g. 'train', 'test', 'dev')

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            l1_labels:
                the frame-wise phone labels how an L1 (native) speaker would
                hear them
                the labels are all padded to the same length

            l2_labels:
                the frame-wise phone labels how an L2 (native) speaker
                intended them
                the labels are all padded to the same length

            l1_lengths:
                list containing the lengths of the unpadded L1 labels

            l2_lengths:
                list containing the lengths of the unpadded L2 labels

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

            logger:
                Tensorboard SummaryWriter, given here for logging/debugging convenience
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

        Return:
            loss:
                the loss to be optimized, should not be detached
        """

        # These can't go in the dataset collate_fn, because they are
        # produced by the upstream model.
        feature_lengths = torch.LongTensor([len(l) for l in features])
        features = pad_sequence(features)

        # We make the predictions with the fine-tuned models using the
        # upstream features
        l1_preds = self.l1_model(features)
        l2_preds = self.l2_model(features)

        # Compute the loss for both models
        l1_loss = self.objective(l1_preds, l1_labels, feature_lengths, l1_lengths)
        l2_loss = self.objective(l2_preds, l2_labels, feature_lengths, l2_lengths)

        loss = l1_loss + l2_loss

        records["loss"].append(loss)

        # Get the true L1 and L2 phones
        true_l1_phones = l1_labels
        true_l2_phones = l2_labels

        # Decode the label sequence from the predictions
        # Note: we are using a greedy decoding algorithm here
        # This is the case, as the non-greedy algos are slow and
        # I have not managed to find algos that run on the GPU that do the decoding
        # TODO: see if there are any CTC decoding algos that run on the GPU
        relative_l1_lengths = [
            l1_len / len(padded_l1_label)
            for (l1_len, padded_l1_label) in zip(l1_lengths, l1_labels)
        ]
        relative_l2_lengths = [
            l2_len / len(padded_l2_label)
            for (l2_len, padded_l2_label) in zip(l2_lengths, l2_labels)
        ]
        pred_l1_phones = ctc_greedy_decode(
            probabilities=rearrange(l1_preds, "L B C -> B C L"),
            seq_lens=relative_l1_lengths,
            blank_id=0,
        )
        pred_l2_phones = ctc_greedy_decode(
            probabilities=rearrange(l2_preds, "L B C -> B C L"),
            seq_lens=relative_l2_lengths,
            blank_id=0,
        )

        # Computing metrics:

        # Compute WER scores for the L1 and L2 predictions
        l1_stats = Counter()
        for true_l1, pred_l1 in zip(true_l1_phones, pred_l1_phones):
            l1_stats = accumulatable_wer_stats(
                refs=true_l1_phones,
                hyps=pred_l1_phones,
                stats=l1_stats,
            )

        # TODO: check if there is a way to store the stats per speaker and aggregate at the end
        #       instead of aggregating per batch
        l2_stats = Counter()
        for true_l2, pred_l2 in zip(true_l2_phones, pred_l2_phones):
            l2_stats = accumulatable_wer_stats(
                refs=true_l2_phones,
                hyps=pred_l2_phones,
                stats=l2_stats,
            )

        records["l1_wer"].append(l1_stats["WER"])
        records["l2_wer"].append(l2_stats["WER"])

        # TODO: verify if this is the right metric to be using
        #       - refer to the papers in the supervisor meeting notes
        # TODO: add a good description of what we are doing
        # Metric for reconstructing L1 and L2 scores: WER
        # TODO: describe it here
        # Metric for mispronunciation deection: f1-score
        # Way of computing it:
        # 1. Finding errors (L1):
        #    - We find the best alignment between the true and predicted L1 phones
        #    - If we match a character in the predictions to one in the true phones,
        #      it's deeded as a correct phone, otherwise, it's a mispronunciation
        # 2. Verifying errors (L2):
        #    - We find the best alignment between the true and predicted L2 phones
        #    - If we match a character in the predictions to one in the true phones,
        #      it's deeded as a phone a person with an accent would understand.
        #      otherwise, it's an error that is **not** the result of mispronunciation.
        # 3. Computing metrics:
        #    - We use F1 score as a metric, where the classes are as follows:
        #       - true positives: the phones understood by the L2 speaker, but not by the L1 speaker
        #       - true negatives: the phones understood by both speakers
        #       - false positives: the phones not understood by anyone
        #       - false negatives: the phones understood by the L1 speaker, but not by the L2 speaker

        l1_pred_stats = wer_details_for_batch(
            ids=range(len(true_l1_phones)),
            refs=true_l1_phones,
            hyps=pred_l1_phones,
            compute_alignments=True,
        )

        l2_pred_stats = wer_details_for_batch(
            ids=range(len(true_l2_phones)),
            refs=true_l2_phones,
            hyps=pred_l2_phones,
            compute_alignments=True,
        )

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i, (l1_pred_stat, l2_pred_stat) in enumerate(
            zip(l1_pred_stats, l2_pred_stats)
        ):
            l1_align = l1_pred_stat["alignment"]
            l2_align = l2_pred_stat["alignment"]

            # This isn't a good assumption, because the length of the alignments
            # depends on the edit distance, so if the two sequences are of
            # different lengths, the length of their alignment sequences will also differ
            # assert len(l1_align) == len(
            # l2_align
            # ), f"Alignment lengths differ: {len(l1_align)} and {len(l2_align)}"

            l1_understood = {
                alignment_tuple[2]
                for alignment_tuple in l1_align
                if alignment_tuple[0] == "="
            }
            l2_understood = {
                alignment_tuple[2]
                for alignment_tuple in l2_align
                if alignment_tuple[0] == "="
            }

            someone_understood = l1_understood | l2_understood

            # Assumption: we have an true error if the L1 speaker
            # wouldn't understand something the L2 speaker would

            # True negative: everyone understands it
            tn += len(l1_understood & l2_understood)
            # True positive: L2 speaker understands, but L1 doesn't
            tp += len(l2_understood - l1_understood)
            # False negative: L1 speaker understands, but L2 doesn't
            fn += len(l1_understood - l2_understood)
            # False positive: No one understands
            fp += len(pred_l1_phones[i]) - len(someone_understood)

        precision = 0
        if tp + fp > 0:
            precision = tp / (tp + fp)
        recall = 0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        f1_score = 0
        if precision != 0 or recall != 0:
            f1_score = 2 / ((1 / precision) + (1 / recall))

        records["precision"].append(precision)
        records["recall"].append(recall)
        records["f1_score"].append(f1_score)

        return loss

    # Interface
    def log_records(
        self, split: str, records: Dict[List[any]], logger, global_step: int, **kwargs
    ):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        # TODO: change avg_loss to some better metric, e.g. accuracy
        prefix = f"mdd/{split}-"

        metrics = [
            "loss",
            "l1_wer",
            "l2_wer",
            "f1_score",
            "precision",
            "recall",
        ]

        message_parts = [
            prefix,
            f"step:{global_step}",
        ]

        avg_loss = torch.FloatTensor(records["loss"]).mean().item()

        for metric in metrics:
            avg_metric = torch.FloatTensor(records[metric]).mean().item()
            logger.add_scalar(f"{prefix}{metric}", avg_metric, global_step=global_step)
            message_parts.append(f"{metric}:{avg_metric}")

        message_parts.append("\n")
        message = "|".join(message_parts)
        save_ckpt = []
        if avg_loss > self.best[prefix]:
            self.best[prefix] = avg_loss
            message = f"best|{message}"
            name = prefix.split("/")[-1].split("-")[0]
            save_ckpt.append(f"best-states-{name}.ckpt")
        with open(self.logging, "a") as f:
            f.write(message)

        return save_ckpt
