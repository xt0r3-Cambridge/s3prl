import abc
import pickle
import logging
from matplotlib.style import available
import numpy as np
from tqdm import tqdm
from enum import Enum
from typing import Any, List
from functools import partial
from dataclasses import dataclass
from inspect import isfunction, ismethod
from joblib import Parallel, delayed

import torch
import torchaudio
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

from speechbrain.utils.data_pipeline import DynamicItem
from speechbrain.dataio.dataset import DynamicItemDataset

from s3prl import Object, cache, Output

logger = logging.getLogger(__name__)


class AugmentedDynamicItemDataset(DynamicItemDataset):
    def __init__(
        self,
        data,
        dynamic_items=[],
        output_keys=[],
        global_stats: dict = {},
    ):
        super().__init__(data, dynamic_items, output_keys)
        self._global_stats = {}
        for name, item in global_stats.keys():
            self.add_global_stats(name, item)

    def _dynamic_global_stats(self, id, name):
        return self._global_stats[name]

    def add_global_stats(self, name: str, item: Any):
        self._global_stats[name] = item
        self.add_dynamic_item(
            partial(self._dynamic_global_stats, name=name), takes="id", provides=name
        )

    def add_dynamic_item(self, func, takes=None, provides=None):
        if isinstance(func, DynamicItem):
            logger.warning(f"Ignoring default takes: {takes}, and provides {provides}")
            takes = func.takes
            provides = func.provides
        super().add_dynamic_item(func, takes, provides)

    def add_dynamic_item_and_metadata(self, func, takes=None, provide=None):
        """
        The function should take `metadata` as an optional argument
        The output dict will auto add a key: f"{provide}_metadata"
        """

        if isinstance(func, DynamicItem):
            logger.warning(f"Ignoring default takes: {takes}, and provide {provide}")
            takes = func.takes
            provide = func.provides
            func = func.func
        assert isfunction(func) or ismethod(func)

        if not isinstance(provide, str):
            assert len(provide) == 1
            provide = provide[0]
        else:
            provide = provide

        self.add_dynamic_item(
            partial(func, metadata=False), takes=takes, provides=provide
        )

        if isinstance(takes, str):
            takes = [takes]

        with self.output_keys_as(["id"] + takes):

            def get_item(item):
                fields = [item[take] for take in takes]
                return fields

            id_to_take_items = {item["id"]: get_item(item) for item in self}

        ids = sorted(id_to_take_items.keys())
        all_take_items = [id_to_take_items[idx] for idx in ids]
        metadatas = self._precompute_metadatas(func, all_take_items)
        id_to_metadata = {idx: metadata for idx, metadata in zip(ids, metadatas)}

        mapping_name = f"_id_to_metadata_for_{provide}"
        self.add_global_stats(mapping_name, id_to_metadata)
        self.add_dynamic_item(
            self._get_metadata,
            takes=["id", mapping_name],
            provides=f"{provide}_metadata",
        )

    @staticmethod
    def _get_metadata(id, mapping):
        return mapping[id]

    @staticmethod
    @cache(signatures=["func", "all_take_items"])
    def _precompute_metadatas(func, all_take_items, n_jobs: int = 8):
        metadatas = Parallel(n_jobs=n_jobs)(
            delayed(partial(func, metadata=True))(*take_items)
            for take_items in tqdm(all_take_items, desc="precompute metadata")
        )
        return metadatas

    @property
    def available_keys(self):
        available_keys: List[str] = list(self.pipeline.key_to_node.keys())
        for dynamic_item in self.pipeline.dynamic_items:
            provides = dynamic_item.provides
            assert isinstance(provides, (list, tuple))
            available_keys += provides
        available_keys = [
            key
            for key in available_keys
            if not key.startswith("_") and key not in self._global_stats
        ]
        return available_keys

    def __getitem__(self, index):
        if len(self.pipeline.output_mapping) == 0:
            with self.output_keys_as(self.available_keys):
                return super().__getitem__(index)
        else:
            return super().__getitem__(index)

    def save_checkpoint(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_checkpoint(cls, path):
        with open(path, "rb") as file:
            obj = pickle.load(file)
        return obj


@dataclass
class DatasetBuilder:
    audio_sample_rate: int = 16000

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, DynamicItem):
            value.func = value.func.__get__(self)
        return value

    def load_audio(self, wav_path, metadata: bool = False):
        if not metadata:
            torchaudio.set_audio_backend("sox_io")
            wav, sr = torchaudio.load(wav_path)
            if sr != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
                wav = resampler(wav)
            wav = wav.view(-1, 1)
            return wav
        else:
            torchaudio.set_audio_backend("sox_io")
            info = torchaudio.info(wav_path)
            ratio = self.audio_sample_rate / info.sample_rate
            return dict(
                sample_rate=self.audio_sample_rate,
                num_frames=round(info.num_frames * ratio),
                num_channels=1,
            )


def default_collate_fn(samples, padding_value: int = 0):
    assert isinstance(samples[0], dict)
    keys = samples[0].keys()
    padded_samples = dict()
    for key in keys:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], (int, float)):
            values = torch.tensor(values)
        elif isinstance(values[0], np.ndarray):
            values = [torch.from_numpy(value) for value in values]
            values = pad_sequence(values, batch_first=True, padding_value=padding_value)
        elif isinstance(values[0], torch.Tensor):
            values = pad_sequence(values, batch_first=True, padding_value=padding_value)
        else:
            values = np.array(values, dtype="object")
        padded_samples[key] = values
    return Output(padded_samples)


class DataLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_sampler,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        collate_fn = collate_fn or default_collate_fn

        super().__init__(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )


class Mode(Enum):
    FULL = 0
    METADATA = 1


_mode = Mode.FULL


def set_mode(mode: str):
    global _mode
    if isinstance(mode, Mode):
        _mode = mode
    elif mode == "FULL":
        _mode = Mode.FULL
    elif mode == "METADATA":
        _mode = Mode.METADATA


def in_metadata_mode():
    return _mode == Mode.METADATA


class metadata_mode:
    def __init__(self):
        self.prev = None

    def __enter__(self):
        self.prev = _mode
        set_mode(Mode.METADATA)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        set_mode(self.prev)


class Dataset(Object, data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def in_metadata_mode():
        return in_metadata_mode()

    @abc.abstractmethod
    def collate_fn(self, samples):
        pass