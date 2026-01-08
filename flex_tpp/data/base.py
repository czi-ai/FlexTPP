from collections import namedtuple
from typing import List, Tuple, TypeVar
from math import log, exp

import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence

MODALITY_CONTINUOUS = 0
MODALITY_CATEGORICAL = 1
MODALITY_ORDINAL = 2
MODALITY_GAUSSIAN = 3

ItemSpec = namedtuple(
    "ItemSpec",
    ["data", "types", "log_prob_correction", "condition", "extras"]
)
BatchSpec = namedtuple(
    "BatchSpec",
    ["data", "types", "log_prob_correction", "condition", "extras"]
)


LOG_EPSILON = 1e-5


T = TypeVar('T')
def log_and_log_abs_det(value: T, epsilon=LOG_EPSILON) -> Tuple[T, T]:
    if torch.is_tensor(value):
        log_value = torch.log(value + epsilon)
    else:
        log_value = log(value + epsilon)
    return log_value, -log_value


def reverse_log(value: T, epsilon=LOG_EPSILON) -> T:
    if torch.is_tensor(value):
        return value.exp() - epsilon
    else:
        return exp(value) - epsilon


def partial_batch(batch: BatchSpec, until_dim: int, overwrite_data: torch.Tensor = None) -> BatchSpec:
    return BatchSpec(
        (overwrite_data if overwrite_data is not None else batch.data)[:, :until_dim],
        batch.types[:, :until_dim],
        None if batch.log_prob_correction is None else batch.log_prob_correction[:, :until_dim],
        batch.condition,
        {
            key: value[:, :until_dim]
            for key, value in batch.extras.items()
        }
    )


def batch_collate(batch: List[ItemSpec]) -> BatchSpec:
    data = pad_sequence([item.data for item in batch], batch_first=True, padding_value=float('nan'))
    types = pad_sequence([item.types for item in batch], batch_first=True, padding_value=MODALITY_CONTINUOUS)

    if batch[0].log_prob_correction is not None:
        log_abs_det = pad_sequence([item.log_prob_correction for item in batch], batch_first=True, padding_value=0.0)
    else:
        log_abs_det = None

    if batch[0].condition is not None:
        condition = pad_sequence([item.condition for item in batch], batch_first=True, padding_value=float('nan'))
    else:
        condition = None

    if batch[0].extras is not None:
        extras = {
            key: pad_sequence(
                [item.extras[key] for item in batch], batch_first=True,
                padding_value=float('nan') if batch[0].extras[key].dtype.is_floating_point else -1
            )
            for key in batch[0].extras
        }
    else:
        extras = None

    return BatchSpec(data, types, log_abs_det, condition, extras)


class FixedDiscretePatternDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, discrete_pattern):
        super().__init__()
        self.dataset = dataset
        self.discrete_pattern = discrete_pattern

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> ItemSpec:
        return ItemSpec(self.dataset[idx], self.discrete_pattern, None, None, None)

    def determine_type_and_extras(self):
        while True:
            for modality in self.discrete_pattern:
                sampled_value = yield modality
                if modality != MODALITY_CONTINUOUS:
                    assert sampled_value.item() == round(sampled_value.item()), f"Asked for discrete value but got floating {sampled_value.item()}"


class BaseDataModule(LightningDataModule):
    def __init__(self, dims: List[int], max_num_classes: int):
        super().__init__()
        self.dims = dims
        self.max_num_classes = max_num_classes
