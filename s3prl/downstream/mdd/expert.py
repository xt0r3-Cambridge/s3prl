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
#   IMPORTS   #
###############
import os
import math
import random
import yaml
from collections import defaultdict, Counter
from typing import Dict, List
from jaxtyping import Float, Int64, jaxtyped
import typeguard

# -------------#
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import ctc_decoder
from torchaudio.functional import edit_distance
from torch.nn.functional import log_softmax
from pathlib import Path
from einops import rearrange
from speechbrain.decoders import ctc_greedy_decode
from speechbrain.utils.edit_distance import (
    wer_details_for_batch,
    accumulatable_wer_stats,
    wer_details_by_utterance,
)

# -------------#
from .model import Model
from .dataset import L2ArcticDataset


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
        self.decoder_params = downstream_expert["decoder_params"]

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

        self.decoder = ctc_decoder(
            **self.decoder_params,
        )

        # Blank is the empty label. In this case, it is the label of silence
        self.objective = nn.CTCLoss(blank=self.arpa_phones["<eps>"], reduction="mean")

        self.logging = Path(expdir) / "log.log"
        self.verbose_log = Path(expdir) / "verbose_log.log"
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
            batch_size=self.datarc[
                f"{split}_batch_size"
            ],  #  - _make_grads in `runner.py` doesn't allow larger batches
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=self.dataset[split].collate_fn,
        )

    def print_phones(self, *args, return_list=False):
        aligned_phones = [[] for i in range(len(args))]

        for phones in zip(*args):
            maxi = max([len(str(phone)) for phone in phones])
            for i, phone in enumerate(phones):
                aligned_phones[i].append(str(phone).center(maxi))

        if return_list:
            return aligned_phones

    def compute_alignment(self, reference, hypothetical, empty_token=-1):
        # TODO: make faster
        dp = [
            [0 for j in range(len(hypothetical) + 1)] for i in range(len(reference) + 1)
        ]
        par = [
            [(-1, -1) for j in range(len(hypothetical) + 1)]
            for i in range(len(reference) + 1)
        ]

        # Initialization of sides of the table
        for i in range(1, len(dp)):
            dp[i][0] = i
            par[i][0] = (i - 1, 0)
        for j in range(1, len(dp[0])):
            dp[0][j] = j
            par[0][j] = (0, j - 1)

        # Computing best path
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                dp[i][j] = 1e9
                # Try eq or sub if possible
                if (
                    dp[i - 1][j - 1] + (reference[i - 1] != hypothetical[j - 1])
                    < dp[i][j]
                ):
                    dp[i][j] = dp[i - 1][j - 1] + (
                        reference[i - 1] != hypothetical[j - 1]
                    )
                    par[i][j] = (i - 1, j - 1)
                # If I or D is better, do that
                if dp[i][j - 1] + 1 < dp[i][j]:
                    dp[i][j] = dp[i][j - 1] + 1
                    par[i][j] = (i, j - 1)
                if dp[i - 1][j] + 1 < dp[i][j]:
                    dp[i][j] = dp[i - 1][j] + 1
                    par[i][j] = (i - 1, j)

        # Retracing path
        path = []
        reference_aligned = []
        hypothetical_aligned = []
        reference_idx = len(reference)
        hypothetical_idx = len(hypothetical)

        while reference_idx > 0 or hypothetical_idx > 0:
            par_reference, par_hypothetical = par[reference_idx][hypothetical_idx]

            op = None

            if (reference_idx - 1, hypothetical_idx - 1) == (
                par_reference,
                par_hypothetical,
            ):
                op = None
                if reference[reference_idx - 1] == hypothetical[hypothetical_idx - 1]:
                    op = "="
                else:
                    op = "S"
                path.append((op, reference_idx - 1, hypothetical_idx - 1))
                reference_aligned.append(reference[reference_idx - 1])
                hypothetical_aligned.append(hypothetical[hypothetical_idx - 1])
            elif (reference_idx - 1, hypothetical_idx) == (
                par_reference,
                par_hypothetical,
            ):
                path.append(("I", reference_idx - 1, None))
                reference_aligned.append(reference[reference_idx - 1])
                hypothetical_aligned.append(empty_token)
            elif (reference_idx, hypothetical_idx - 1) == (
                par_reference,
                par_hypothetical,
            ):
                path.append(("D", None, hypothetical_idx - 1))
                reference_aligned.append(empty_token)
                hypothetical_aligned.append(hypothetical[hypothetical_idx - 1])
            else:
                raise IndexError(
                    f"Invalid parents for index ({reference_idx}, {hypothetical_idx}): ({par_reference}, {par_hypothetical})"
                )

            reference_idx, hypothetical_idx = par_reference, par_hypothetical

        path = list(reversed(path))
        reference_aligned = list(reversed(reference_aligned))
        hypothetical_aligned = list(reversed(hypothetical_aligned))

        assert len(reference_aligned) == len(
            hypothetical_aligned
        ), f"""Unequal lengths detected for the aligned sequences
reference: {len(reference_aligned)}
hypothetical: {len(hypothetical_aligned)}
"""

        return {
            "edit_distance": dp[-1][-1],
            "alignment": path,
            "reference_aligned": reference_aligned,
            "hypothetical_aligned": hypothetical_aligned,
        }

    # @jaxtyped
    # @typeguard.typechecked
    def compute_asr_metrics(
        self,
        true_canonical_phones: List[Int64[Tensor, "seq_len1"]],
        pred_canonical_phones: List[Int64[Tensor, "seq_len2"]],
    ) -> Counter:
        """
        Computes the ASR metrics for the phone reconstruction of our model.

        Returns:
            collections.Counter
            The updated running statistics, with keys:

            "WER" - word error rate
            "insertions" - number of insertions
            "deletions" - number of deletions
            "substitutions" - number of substitutions
            "num_ref_tokens" - number of reference tokens
        """
        counter = Counter()
        prediction_error_metrics = accumulatable_wer_stats(
            refs=true_canonical_phones,
            hyps=pred_canonical_phones,
            stats=counter,
        )
        return prediction_error_metrics

    @jaxtyped
    @typeguard.typechecked
    def compute_mdd_metrics(
        self,
        true_perceived_phones: Int64[Tensor, "seq_len"],
        true_canonical_phones: Int64[Tensor, "seq_len2"],
        pred_canonical_phones: Int64[Tensor, "seq_len3"],
    ):
        """
        TODO: this only works for a single batch
        TODO: make faster

        Computes precision, recall and F1-score for the mispronunciation detection model.
        All of these need the following 4 metrics:
        - number of true positives
        - number of true negatives
        - number of false positives
        - number of false negatives

        We compute these as follows:
        1. We align the sequences (using edit-distance alignment) in the following ways:
            - align true perceieved and true canonical phones (true alignment)
            - align true perceieved and predicted canonical phones (pred alignment)
            - pad all of the alignments with empty "<eps>" characters such that all three strings are aligned
        2. Using the alignments, we compute the errors:
            - true positives: neither the true or the predicted canonical phone agree with the true perceived phone
            - true negatives: both the true and the predicted canonical phone agree with the true perceived phone
            - false positives: the true canonical and true perceived phones are the same, but the predicted canonical phone differs
            - false negatives: the predicted canonical and true perceived phones are the same, but the true canonical phone differs
        """
        empty_token = -1
        true_alignment = self.compute_alignment(
            reference=true_canonical_phones,
            hypothetical=true_perceived_phones,
            empty_token=empty_token,
        )

        pred_alignment = self.compute_alignment(
            reference=true_canonical_phones,
            hypothetical=pred_canonical_phones,
            empty_token=empty_token,
        )

        # Pad the alignments with empty characters so that they are all aligned
        aligned_true_canonical_phones = []
        aligned_true_perceived_phones = []
        aligned_pred_canonical_phones = []

        true_alignment_idx = 0
        pred_alignment_idx = 0

        true_reference = true_alignment["reference_aligned"]
        pred_reference = pred_alignment["reference_aligned"]
        true_hypothetical = true_alignment["hypothetical_aligned"]
        pred_hypothetical = pred_alignment["hypothetical_aligned"]

        # TODO: write tests to make sure this is correct
        while true_alignment_idx < len(true_reference) and pred_alignment_idx < len(
            pred_reference
        ):
            if true_reference[true_alignment_idx] == pred_reference[pred_alignment_idx]:
                aligned_true_canonical_phones.append(true_reference[true_alignment_idx])
                aligned_pred_canonical_phones.append(
                    pred_hypothetical[pred_alignment_idx]
                )
                aligned_true_perceived_phones.append(
                    true_hypothetical[true_alignment_idx]
                )
                true_alignment_idx += 1
                pred_alignment_idx += 1
            elif true_reference[true_alignment_idx] == empty_token:
                aligned_true_canonical_phones.append(empty_token)
                aligned_pred_canonical_phones.append(empty_token)
                aligned_true_perceived_phones.append(
                    true_hypothetical[true_alignment_idx]
                )
                true_alignment_idx += 1
            elif pred_reference[pred_alignment_idx] == empty_token:
                aligned_true_canonical_phones.append(empty_token)
                aligned_pred_canonical_phones.append(
                    pred_hypothetical[pred_alignment_idx]
                )
                aligned_true_perceived_phones.append(empty_token)
                pred_alignment_idx += 1
            else:
                assert False, f"""Invalid alignment parse for strings:
true_reference: {true_reference}
pred_reference: {pred_reference}
true_hypothetical: {true_hypothetical}
pred_hypothetical: {pred_hypothetical}

Exiting...
"""

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        corr_diag = 0
        err_diag = 0

        for true_canonical_phone, pred_canonical_phone, true_perceived_phone in zip(
            aligned_true_canonical_phones,
            aligned_pred_canonical_phones,
            aligned_true_perceived_phones,
        ):
            if true_canonical_phone != true_perceived_phone:
                if pred_canonical_phone != true_perceived_phone:
                    tp += 1
                    if pred_canonical_phone == true_canonical_phone:
                        corr_diag += 1
                    else:
                        err_diag += 1
                else:
                    fn += 1
            else:
                if pred_canonical_phone != true_perceived_phone:
                    fp += 1
                else:
                    tn += 1

        precision = 0
        if tp + fp > 0:
            precision = tp / (tp + fp)
        recall = 0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        f1_score = 0
        if precision != 0 or recall != 0:
            f1_score = 2 / ((1 / precision) + (1 / recall))

        corr_diag_pct = corr_diag / tp if tp > 0 else 0
        err_diag_pct = err_diag / tp if tp > 0 else 0

        aligned_list = self.print_phones(
            aligned_pred_canonical_phones,
            aligned_true_canonical_phones,
            aligned_true_perceived_phones,
            return_list=True,
        )
        with open(self.verbose_log, "a") as f:
            f.writelines([" | ".join(l) + "\n" for l in aligned_list])
            f.write("\n\n")

        return {
            "corr_diag_pct": corr_diag_pct,
            "err_diag_pct": err_diag_pct,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
    
    # Interface
    """runner.py releavant code:
        loss = self.downstream.model(
            train_split,
            features, *others,
            records = records,
        )
    """

    @jaxtyped
    @typeguard.typechecked
    def forward(
        self,
        split: str,
        features: List[Tensor],
        true_perceived_phones: Int64[Tensor, "batch seq_len"],
        true_canonical_phones: Int64[Tensor, "batch seq_len2"],
        perceived_lengths: Int64[Tensor, "batch"],
        canonical_lengths: Int64[Tensor, "batch"],
        records: any,
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
        # print(features)
        feature_lengths = torch.LongTensor([len(l) for l in features])
        features = pad_sequence(features, padding_value=self.arpa_phones["<eps>"])

        # We make the predictions with the fine-tuned models using the
        # upstream features
        pred_phone_logits = self.mdd_model(features)
        pred_phone_distribution = log_softmax(pred_phone_logits, dim=-1)

        # Compute the CTC loss
        loss = self.objective(
            pred_phone_distribution,
            true_canonical_phones,
            feature_lengths,
            canonical_lengths,
        )

        records["loss"].append(loss)

        if split == "train":
            return loss

        # Get the true L1 and L2 phones

        # Decode the label sequence from the predictions
        # Note: we are using a greedy decoding algorithm here
        # This is the case, as the non-greedy algos are slow and
        # I have not managed to find algos that run on the GPU that do the decoding

        # TODO: see if there are any CTC decoding algos that run on the GPU
        with torch.no_grad():
            # TODO: if this fails working, change this back to a log_softmax version
            rearranged_predictions = rearrange(
                pred_phone_distribution, "L B C -> B L C"
            )
            pred_sequences = self.decoder(
                emissions=rearranged_predictions.cpu().contiguous(),
                # TODO: test if correct -- potentially change
                lengths=feature_lengths,
                # seq_lens=torch.ones(rearranged_predictions.shape[0]),
                # blank_id=self.arpa_phones["<eps>"],
            )

            # Remove multiple silences -- we know this can never occur in the dataset
            pred_canonical_phones = [torch.LongTensor(self.dataset[split].remove_multiple_sil(seq[0].tokens)) for seq in pred_sequences]

            # TODO: check if there is a way to store the stats per speaker and aggregate at the end
            #       instead of aggregating per batch

            # Compute WER scores for the canonical label reconstruction
            # This is an ASR task, so we use the WER metric to compute the phoneme error rate
            asr_metrics = self.compute_asr_metrics(
                # TODO: check if the [:canonical_lenghts] is correct
                true_canonical_phones=true_canonical_phones[:canonical_lengths],
                pred_canonical_phones=pred_canonical_phones,
            )
            records["per"].append(asr_metrics["WER"])

            # TODO: refer to the paper we took this approach from
            # https://arxiv.org/pdf/2203.15937.pdf

            for (
                true_canonical_1batch,
                true_perceived_1batch,
                pred_canonical_1batch,
                canonical_length_1batch,
                perceived_length_1batch,
            ) in zip(
                true_canonical_phones,
                true_perceived_phones,
                pred_canonical_phones,
                canonical_lengths,
                perceived_lengths,
            ):
                mdd_metrics = self.compute_mdd_metrics(
                    # TODO: check if the [:canonical/perceibed_lenght_1batch] is correct
                    true_canonical_phones=true_canonical_1batch[
                        :canonical_length_1batch
                    ],
                    true_perceived_phones=true_perceived_1batch[
                        :perceived_length_1batch
                    ],
                    pred_canonical_phones=pred_canonical_1batch,
                )

                for k, v in mdd_metrics.items():
                    records[k].append(v)

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
            "corr_diag_pct",
            "err_diag_pct",
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
            if metric in records:
                avg_metric = torch.FloatTensor(records[metric]).mean().item()
                logger.add_scalar(
                    f"{prefix}{metric}", avg_metric, global_step=global_step
                )
                message_parts.append(f"{metric}:{avg_metric}")

        message_parts.append("\n")
        message = "|".join(message_parts)

        save_ckpt = []
        if avg_loss < self.best[prefix]:
            self.best[prefix] = avg_loss
            message = f"best|{message}"
            name = prefix.split("/")[-1].split("-")[0]
            save_ckpt.append(f"best-states-{name}.ckpt")
        with open(self.logging, "a") as f:
            f.write(message)

        return save_ckpt
