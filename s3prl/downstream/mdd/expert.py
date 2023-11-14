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
from collections import defaultdict

# -------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import ctc_decoder
from torchaudio.functional import edit_distance
from pathlib import Path
from einops import rearrange

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

        with open(Path(self.datarc['phone_path']) / "tokens.txt", "r") as stream:
            self.arpa_phones = {phone.strip(): i for i, phone in enumerate(stream.readlines())}

        # Blank is the empty label. In this case, it is the label of silence
        self.objective = nn.CTCLoss(blank=self.arpa_phones["sil"], reduction="none")

        # Create mapping from 
        self.decoder = ctc_decoder(
            lexicon=None,
            tokens=f"{self.datarc['phone_path']}/tokens.txt",
            lm=None,  # no language model used
            lm_dict = None, 
            nbest=1, # beam search returns top 1 best result
            beam_size=50, # we retain 50 beams during beam search
            beam_size_token=None,
            beam_threshold=50,
            lm_weight=2,
            word_score=0,
            unk_score=-float('inf'),
            sil_score=0,
            log_add=False,
            blank_token='sil',
            sil_token='sil',
            unk_word='<unk>',
        )


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
        split,
        features,
        l1_labels,
        l2_labels,
        l1_lengths,
        l2_lengths,
        records,
        **kwargs,
    ):
        """
        TODO: rewrite this
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

                TODO: figure out if these are all the same length

            labels:
                the frame-wise phone labels

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

            logger:
                Tensorboard SummaryWriter, given here for logging/debugging convenience
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging

        Return:
            loss:
                the loss to be optimized, should not be detached
        """

        # These can't go in the dataset collate_fn, because they are
        # produced by the upstream model.
        feature_lengths = torch.LongTensor([len(l) for l in features])
        features = pad_sequence(features)

        l1_preds = self.l1_model(features)
        l2_preds = self.l2_model(features)

        # print(f'feature shape: {features.shape}')
        # print(feature_lengths.shape)
        # print(l1_preds.shape)

        l1_loss = self.objective(l1_preds, l1_labels, feature_lengths, l1_lengths)
        l2_loss = self.objective(l2_preds, l2_labels, feature_lengths, l2_lengths)

        loss = l1_loss + l2_loss

        # TODO: maybe add logging
        records["loss"].append(loss)

        pred_l1_phones = self.decoder(rearrange(l1_preds.to('cpu'), 'L B C -> B L C'))
        true_l1_phones = l1_labels
        pred_l2_phones = self.decoder(rearrange(l2_preds.to('cpu'), 'L B C -> B L C'))
        true_l2_phones = l2_labels

        # print(pred_l1_phones[0])


        for pred_l1, true_l1, len_l1, pred_l2, true_l2, len_l2 in zip(pred_l1_phones, true_l1_phones, l1_lengths, pred_l2_phones, true_l2_phones, l2_lengths):
            assert len_l1 == len_l2, f"Length mismatch between l1 and l2 labels: {len_l1} and {len_l2}"
            pred_l1 = pred_l1[0].timesteps
            pred_l2 = pred_l2[0].timesteps
            # print(pred_l1.shape)
            # print(true_l1.shape)


            # records['l1_acc'].append((pred_l1[:len_l1] == true_l1[:len_l1]).sum() / len_l1)
            records['l1_wer'].append(edit_distance(true_l1[:len_l1], pred_l1[:len_l1]) / len_l1)
            # records['l2_acc'].append((pred_l2[:len_l2] == true_l2[:len_l2]).sum() / len_l2)
            records['l2_wer'].append(edit_distance(true_l2[:len_l2], pred_l2[:len_l2]) / len_l1)
            # Problem: 
            # pred_mismatch = pred_l1_phones != pred_l2_phones
            # true_mismatch = true_l1_phones != true_l2_phones
            # records['mdd_acc'].append((pred_mismatch[:len_l1] == true_mismatch[:len_l2]).sum() / len_l1)
            # records['mdd_wer'].append(edit_distance(pred_mismatch[:len_l1], true_mismatch[:len_l2]) / len_l1)

        # TODO: correctly decode l1_preds using a CTC decoder
        # predicted_mismatch = l1_preds.max(dim=-1).indices != l2_preds.max(dim=-1).indices
        # true_mismatch = l1_labels != l2_labels
        # sames = predicted_classid == labels
        # for s, l in zip(sames, lengths):
        #     utter_result = s[:l].tolist()
        #     records["acc"] += utter_result
        #     records["sample_wise_metric"] += [
        #         torch.FloatTensor(utter_result).mean().item()
        #     ]

        return loss

    # interface
    def log_records(self, split, records, logger, global_step, **kwargs):
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
        avg_loss = torch.FloatTensor(records["loss"]).mean().item()
        # avg_l1_acc = torch.FloatTensor(records["l1_acc"]).mean().item()
        # avg_l2_acc = torch.FloatTensor(records["l2_acc"]).mean().item()
        avg_l1_wer = torch.FloatTensor(records["l1_wer"]).mean().item()
        avg_l2_wer = torch.FloatTensor(records["l2_wer"]).mean().item()
        # avg_mdd_acc = torch.FloatTensor(records["mdd_acc"]).mean().item()
        # avg_mdd_wer = torch.FloatTensor(records["mdd_wer"]).mean().item()

        logger.add_scalar(f"{prefix}loss", avg_loss, global_step=global_step)
        # logger.add_scalar(f"{prefix}l1_acc", avg_l1_acc, global_step=global_step)
        logger.add_scalar(f"{prefix}l1_wer", avg_l1_wer, global_step=global_step)
        # logger.add_scalar(f"{prefix}l2_acc", avg_l2_acc, global_step=global_step)
        logger.add_scalar(f"{prefix}l2_wer", avg_l2_wer, global_step=global_step)
        # logger.add_scalar(f"{prefix}mdd_acc", avg_mdd_acc, global_step=global_step)
        # logger.add_scalar(f"{prefix}mdd_wer", avg_mdd_wer, global_step=global_step)
        # message = f"{prefix}|step:{global_step}|loss:{avg_loss}|mdd_acc:{avg_mdd_acc}|mdd_wer:{avg_mdd_wer}|l1_acc:{avg_l1_acc}|l1_wer:{avg_l1_wer}|l2_acc:{avg_l2_acc}|l2_wer:{avg_l2_wer}|\n"
        # message = f"{prefix}|step:{global_step}|loss:{avg_loss}|mdd_wer:{avg_mdd_wer}|l1_wer:{avg_l1_wer}|l2_wer:{avg_l2_wer}|\n"
        message = f"{prefix}|step:{global_step}|loss:{avg_loss}|l1_wer:{avg_l1_wer}|l2_wer:{avg_l2_wer}|\n"
        save_ckpt = []
        if avg_loss > self.best[prefix]:
            self.best[prefix] = avg_loss
            message = f"best|{message}"
            name = prefix.split("/")[-1].split("-")[0]
            save_ckpt.append(f"best-states-{name}.ckpt")
        with open(self.logging, "a") as f:
            f.write(message)

        return save_ckpt
