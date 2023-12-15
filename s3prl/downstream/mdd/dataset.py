# Code for: Loading TIMIT for linear phoneme recognition
# Goal: to get something that loads the L2Arctic dataset


# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ Loading the L2Arctic dataset for mispronunciation detection and diagnosis (MDD) ]
#   Author       [ Kornel Szabo ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random
import yaml

# -------------#
import pandas as pd

# -------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from textgrid import TextGrid, IntervalTier
from pathlib import Path

# -------------#

import torchaudio


HALF_BATCHSIZE_TIME = 2000


#####################
# L2 Arctic Dataset #
#####################
class L2ArcticDataset(Dataset):
    def __init__(
        self,
        split,
        # bucket_size,
        data_root,
        phone_path,
        # bucket_file,
        sample_rate=44100,
        train_dev_seed=1337,
        **kwargs,
    ):
        super().__init__()

        assert split in [
            "train",
            "dev",
            "test",
        ], f"Invalid split '{split}'. Expected 'train', 'dev' or 'test'"

        self.data_root = data_root
        self.sample_rate = sample_rate

        with open(Path(phone_path) / "dataset_config.yaml", "r") as stream:
            self.config = yaml.safe_load(stream)

        with open(Path(phone_path) / "tokens.txt", "r") as stream:
            self.arpa_phones = {
                phone.strip(): i for i, phone in enumerate(stream.readlines())
            }

        # The number of different phones
        self.output_class_num = max(self.arpa_phones.values()) + 1

        self.item_paths = []
        for speaker in self.config[split]["speakers"]:
            item_root = Path(self.data_root) / speaker
            for annotation_file in (item_root / "annotation").iterdir():
                # Only save files that are formatted well
                # We expect two malformatted files
                # - YDCK/annotation/arctic_a0272.TextGrid
                # - YDCK/annotation/arctic_a0209.TextGrid
                tg = TextGrid()
                try:
                    tg.read(annotation_file)
                    self.item_paths.append(
                        {"item_root": item_root, "item_stem": annotation_file.stem}
                    )
                except Exception as e:
                    print(f"Skipping malformatted TextGrid file: {annotation_file}")

        # TODO: remove -- added to test if we can overfit
        # self.item_paths = self.item_paths[:32]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.data_root, wav_path))
        assert (
            sr == self.sample_rate
        ), f"Sample rate mismatch: real {sr}, config {self.sample_rate}"
        return wav.view(-1)

    def _strip_arpa_phone(self, phone):
        """
        Strips stress and auxilliary symbols from an ARPA phone.
        Detailed description of ARPA phones [here](https://en.wikipedia.org/wiki/ARPABET)

        E.g. the ARPA code AA1 means the phone AA with primary stress

        For this dataset, this is equivalent to just taking the longest prefix that only contains alphanumeric characters

        Note that capitalisation differs across the dataset, so we always just take the lowercase variant
        (see TLV/annotation/arctic_b0534.TextGrid for 'l,sil,d')
        """
        # Special case:
        # in case of silence, sometimes
        # "" is used instead of "sil"
        phone = phone.strip().lower()
        # Silence is either denoted by 'sp' or by 'sil' or '' or 'spn'
        if phone in ["sil", "", "sp", "spn"]:
            return "sil"
        # nx and ng are the same sound
        elif phone in ["nx", "ng"]:
            return "nx"
        # h and hh are the same sound
        elif phone in ["h", "hh"]:
            return "hh"
        substr = ""
        for c in phone:
            if c.isalpha():
                substr = substr + c
            else:
                break
        return substr

    def __len__(self):
        # TODO: change
        return len(self.item_paths)

    def __getitem__(self, index):
        item_root = self.item_paths[index]["item_root"]
        item_stem = self.item_paths[index]["item_stem"]

        wav_view = self._load_wav(item_root / f"wav/{item_stem}.wav")

        annotation_file = item_root / f"annotation/{item_stem}.TextGrid"
        # Read the TextGrid file
        # An example datapoint can be found here:
        # https://psi.engr.tamu.edu/wp-content/uploads/2018/03/lxc_arctic_a0018.txt
        tg = TextGrid()
        try:
            tg.read(annotation_file)
        except Exception as e:
            print(f"Problematic file: {annotation_file}")
            raise e

        canonical = []
        perceived = []

        # This returns the `IntervalTier` which has `name == 'phones'`
        # This contains the intervals which each have a start time, and end time and a mark
        annotation_tier = tg.getFirst("phones")
        for annotation_interval in annotation_tier:
            # An annotation can be e.g.
            # - AA1,AH1,s meaning that
            #    - AA1 is the canonical phone
            #    - AH1 is the perceived phone
            #    - the speaker name a substitution (s) error
            # - AA meaning that AA is the correct and perceived phone

            # Edge cases:
            # - TNI/annotation/arctic_a0053.TextGrid - ' ' as an annotation
            # - TLV/annotation/arctic_b0534.TextGrid - 'l,sil,d' instead of 'L,sil,d'
            # - YDCK/annotation/arctic_a0272.TextGrid - xmax of 'IPA' is out of xmax of file
            annotation = annotation_interval.mark
            phones = annotation.split(",")
            canonical_phone = phones[0]
            perceived_phone = canonical_phone
            # TODO: decide what we want to do in case of 'err'
            # 'err' means that the correct phoneme was unintelligible
            if len(phones) > 1 and self._strip_arpa_phone(phones[1]) != "err":
                perceived_phone = phones[1]

            try:
                canonical.append(
                    self.arpa_phones[self._strip_arpa_phone(canonical_phone)]
                )
                perceived.append(
                    self.arpa_phones[self._strip_arpa_phone(perceived_phone)]
                )
            except Exception as e:
                print(f"Problematic file: {annotation_file}")
                print(f"Problematic label: {repr(annotation)}")
                raise e

            assert len(canonical) == len(
                perceived
            ), f"""Error: canonical and perceived phoneme sequences have differing lengths:
canonical: {canonical}
perceived: {perceived}
"""

        # `runner.py` expects the returned items to be of form (wav_view, *others)
        return (
            wav_view,
            torch.LongTensor(canonical),
            torch.LongTensor(perceived),
            len(canonical),
            len(perceived),
        )

    def collate_fn(self, items):
        wav_views = []
        canonicals = []
        perceiveds = []
        canonical_lengths = []
        perceived_lengths = []
        for wav_view, canonical, perceived, canonical_length, perceived_length in items:
            wav_views.append(wav_view)
            canonicals.append(canonical)
            perceiveds.append(perceived)
            canonical_lengths.append(canonical_length)
            perceived_lengths.append(perceived_length)

        # We don't convert to tensors because the wav_views are
        # of different lengths.
        # Note that the wav_views are not consumed by the downstrea model but rather
        # the upstream one, which expects them to be unpadded
        return (
            wav_views,
            pad_sequence(canonicals, batch_first=True),
            pad_sequence(perceiveds, batch_first=True),
            torch.LongTensor(canonical_lengths),
            torch.LongTensor(perceived_lengths),
        )
