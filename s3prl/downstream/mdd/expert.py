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
        """
        @param upstream_dim: the dimensions of the upstream model
        @param downstream_expert: the `downstream_expert` part of the config file loaded from `./config.yaml`
        @param expdir: the directory to save the experiment in
        """
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
        self.mdd_model = Model(
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

        self.logging = Path(expdir) / "log.log"
        self.best = defaultdict(lambda: 0)

    # Interface
    def get_dataloader(self, split):
        """
        Dataloader Specs:
            Each dataloader should output in the following format:

            (
                [wav1, wav2, ...],
                [perceived_label_1, perceived_label_2, ...],
                [canonical_label_1, canonical_label_2, ...],
            )

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==44100
            and
            perceived_label_1, ... are LongTensors with the encodings of the target label
            sequence
        """
        assert split in [
            "train",
            "dev",
            "test",
        ], f"Invalid split {split}. Expected 'train', 'dev' or 'test'."
        return DataLoader(
            self.dataset[split],
            batch_size=1,  # self.datarc[f'{split}_batch_size'],  - _make_grads in `runner.py` doesn't allow larger batches
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
        true_perceived_phones: torch.FloatTensor,
        true_canonical_phones: torch.FloatTensor,
        perceived_lengths: torch.LongTensor,
        canonical_lengths: torch.LongTensor,
        records: Dict[str, List[any]],
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

            true_perceived_phones:
                the frame-wise phone labels how an L1 (native) speaker would
                hear them
                the labels are all padded to the same length

            true_canonical_phones:
                the frame-wise phone labels how an L2 (native) speaker
                intended them
                the labels are all padded to the same length

            perceived_lengths:
                list containing the lengths of the unpadded perceived labels

            canonical_lengths:
                list containing the lengths of the unpadded canonical labels

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
        pred_phone_distributions = self.mdd_model(features)

        # Compute the CTC loss
        loss = self.objective(
            pred_phone_distributions,
            true_canonical_phones,
            feature_lengths,
            canonical_lengths,
        )
        records["loss"].append(loss)

        # Get the true L1 and L2 phones

        # Decode the label sequence from the predictions
        # Note: we are using a greedy decoding algorithm here
        # This is the case, as the non-greedy algos are slow and
        # I have not managed to find algos that run on the GPU that do the decoding

        # TODO: see if there are any CTC decoding algos that run on the GPU
        relative_canonical_lengths = [
            canonical_length / len(padded_true_canonical_phones)
            for (canonical_length, padded_true_canonical_phones) in zip(
                canonical_lengths, true_canonical_phones
            )
        ]

        pred_canonical_phones = ctc_greedy_decode(
            probabilities=rearrange(pred_phone_distributions, "L B C -> B C L"),
            seq_lens=relative_canonical_lengths,
            blank_id=0,
        )

        # TODO: check if there is a way to store the stats per speaker and aggregate at the end
        #       instead of aggregating per batch

        # Compute WER scores for the predictions
        prediction_error_metrics = Counter()
        prediction_error_metrics = accumulatable_wer_stats(
            refs=true_canonical_phones,
            hyps=pred_canonical_phones,
            stats=prediction_error_metrics,
        )

        # Phoneme error rate
        records["per"].append(prediction_error_metrics["WER"])

        # TODO: refer to the paper we took this approach from

        # Metric for mispronunciation deection: f1-score
        # Way of computing it:
        # 1. Aligning the true and predicted canonical sequences
        # 2. Using the true perceived sequences to find the errors:
        #    - true positives: neither the true or the predicted canonical phone agree with the true perceived phone
        #    - true negatives: both the true and the predicted canonical phone agree with the true perceived phone
        #    - false positives: the true canonical and true perceived phones are the same, but the predicted canonical phone differs
        #    - false negatives: the predicted canonical and true perceived phones are the same, but the true canonical phone differs

        prediction_stats = wer_details_for_batch(
            ids=range(len(true_canonical_phones)),
            refs=true_canonical_phones,
            hyps=pred_canonical_phones,
            compute_alignments=True,
        )

        empty_token = "<eps>"
        for prediction_stat, perceived_utterance in zip(
            prediction_stats, true_perceived_phones
        ):
            # We assume that there is no need for aligning the L1 and L2 labels,
            # because the minimum edit distance between them can be achieved using
            # only substitutions and matches. This holds for all data points,
            # except data point #2006 in our dataset.
            # Data point #2006 has an edit distance of 10 instead of 9 if using this method.
            # We choose to use this simplification and potentially remove data point #2006 in
            # the future.
            pred_to_true_canonical_alignment = prediction_stat["alignment"]
            canonical_utterance = prediction_stat["ref_tokens"]
            pred_canonical_utterance = prediction_stat["hyp_tokens"]

            tp = 0
            fp = 0
            tn = 0
            fn = 0

            for op, true_idx, pred_idx in pred_to_true_canonical_alignment:
                perceived_phone = (
                    perceived_utterance[true_idx]
                    if true_idx is not None
                    else empty_token
                )
                canonincal_phone = (
                    canonical_utterance[true_idx]
                    if true_idx is not None
                    else empty_token
                )
                pred_canonincal_phone = (
                    pred_canonical_utterance[pred_idx]
                    if pred_idx is not None
                    else empty_token
                )
                if op == "=":  # pred_canonical == true_canonical
                    if perceived_phone == canonincal_phone:
                        tn += 1
                    else:
                        tp += 1
                elif op == "S":  # substitution
                    # This still detects mispronunciation
                    if (
                        canonincal_phone != perceived_phone
                        and pred_canonincal_phone != perceived_phone
                    ):
                        tp += 1
                    # We fail to detect mispronuncaition
                    elif (
                        canonincal_phone != perceived_phone
                        and pred_canonincal_phone == perceived_phone
                    ):
                        fn += 1
                    # We incorrectly detect mispronuncation
                    elif (
                        canonincal_phone == perceived_phone
                        and pred_canonincal_phone != perceived_phone
                    ):
                        fp += 1
                    else:
                        assert False, "Incprrect substitution case"
                elif op == "I":  # insertion
                    assert true_idx == None, "Incorrect true index for insertion"
                    # We think an insertion error occurred, but in reality it was a substitution error
                    if canonincal_phone != perceived_phone:
                        tp += 1
                    # We think an insertion error occurred, but nothing did
                    else:
                        fp += 1
                elif op == "D":  # deletion
                    assert pred_idx == None, "Incorrect pred index for deletion"
                    # We think a deletion error occurred, but in reality it was a substitution error
                    if canonincal_phone != perceived_phone:
                        tp += 1
                    # We think a deletion error occurred, but nothing did
                    else:
                        fp += 1

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
        self,
        split: str,
        records: Dict[str, List[any]],
        logger,
        global_step: int,
        **kwargs,
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
        prefix = f"mdd/{split}-"

        metrics = [
            "loss",
            "per",
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
