from typing import Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from flex_tpp.data.base import BaseDataModule, MODALITY_CONTINUOUS, MODALITY_CATEGORICAL, ItemSpec, \
    batch_collate, BatchSpec
from torch.utils.data import DataLoader

from flex_tpp.data.base import log_and_log_abs_det, reverse_log


def get_max_event_type(dataset):
    event_types = set()
    for sequence in dataset['train']:
        event_types.update(sequence["type_event"])
    return max(event_types)


class EventDataset(Dataset):
    DEFAULT_ORDER = "ST"

    def __init__(self, data, order=DEFAULT_ORDER):
        self.data = data
        self.order = order

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        event_vector_parts = []
        log_abs_det_vector_parts = []
        type_vector_parts = []
        position_vector_parts = []
        event_index_vector_parts = []
        for event_idx, (event_time, event_type) in enumerate(
                zip(sample['time_since_last_event'], sample['type_event'])
        ):
            start_time_log, start_time_vol_change = log_and_log_abs_det(event_time)
            for position, entry in enumerate(self.order):
                if entry == "S":
                    data, log_prob_correction, dtypes = [start_time_log], [start_time_vol_change], [MODALITY_CONTINUOUS]
                elif entry == "T":
                    data, log_prob_correction, dtypes = [event_type], [0], [MODALITY_CATEGORICAL]
                else:
                    raise ValueError(f"Unknown event type {entry}")
                event_vector_parts += data
                log_abs_det_vector_parts += log_prob_correction
                type_vector_parts += dtypes
                position_vector_parts += range(position, position + len(data))
                event_index_vector_parts += [event_idx] * len(data)
        data = torch.tensor(event_vector_parts)
        if data.dtype != torch.float32:
            raise ValueError(data.dtype)

        return ItemSpec(
            data,
            torch.tensor(type_vector_parts),
            torch.tensor(log_abs_det_vector_parts).float(),
            None,
            {
                "position_in_event": torch.tensor(position_vector_parts),
                "event_index": torch.tensor(event_index_vector_parts),
            }
        )

    @torch.no_grad()
    def validation_metrics(self, batch: BatchSpec, model, argmax=True, mean_of=50, median=False) -> Dict[str, float]:
        log_prob, parameters = model.log_prob(batch, return_parameters=True)
        # Parallel sampling of all next items in vector
        x_out = torch.full_like(batch.data, float("nan"))
        batch_types = batch.types
        for modality, sample_fn in model.sample_functions(argmax, mean_of, median=median).items():
            modality_mask = (batch_types == modality) & torch.isfinite(batch.data)
            # noinspection PyTypeChecker
            if torch.any(modality_mask):
                modality_parameters = torch.full(
                    (*batch.data.shape, parameters[modality].shape[-1]), float("nan"),
                    device=parameters[modality].device, dtype=parameters[modality].dtype
                )
                modality_parameters[modality_mask] = parameters[modality]
                x_out[modality_mask] = sample_fn(modality_parameters[modality_mask]).to(x_out)

        # Parse prediction according to the order
        metrics = {}
        if "T" in self.order:
            event_type_offset = self.order.index("T")
            event_type_batch = batch.data[torch.isfinite(batch.data)][event_type_offset::2]
            event_type_predicted = x_out[torch.isfinite(x_out)][event_type_offset::2]
            metrics["error_rate"] = 1 - (event_type_batch == event_type_predicted).float().mean()
        if "S" in self.order:
            # Parse prediction according to the order
            relative_time_offset = self.order.index("S")
            relative_time_batch = reverse_log(batch.data[torch.isfinite(batch.data)][relative_time_offset::2])
            relative_time_predicted = reverse_log(x_out[torch.isfinite(x_out)][relative_time_offset::2])
            metrics["rmse"] = ((relative_time_batch - relative_time_predicted) ** 2).float().mean().sqrt()
        return metrics


class TPPDataModule(BaseDataModule):
    def __init__(self, dim, max_num_classes, dataset='taxi',
                 order=EventDataset.DEFAULT_ORDER, **kwargs):
        super().__init__([dim], max_num_classes)
        self.name = dataset

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.order = order
        self.kwargs = kwargs

    def setup(self, stage=None):
        dataset = load_dataset(f"easytpp/{self.name}", revision="main")
        self.train_dataset = EventDataset(dataset['train'], order=self.order)
        self.val_dataset = EventDataset(dataset['validation'], order=self.order)
        self.test_dataset = EventDataset(dataset['test'], order=self.order)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, collate_fn=batch_collate, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, collate_fn=batch_collate, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, collate_fn=batch_collate, **self.kwargs)
