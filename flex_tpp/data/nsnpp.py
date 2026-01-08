
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from flex_tpp.data.base import BaseDataModule, ItemSpec, MODALITY_CONTINUOUS, batch_collate
from flex_tpp.data.property_mtpp import log_and_log_abs_det


class NSPPPDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        times = self.data[idx]
        time_diffs = torch.from_numpy(np.diff(times[times > 0], prepend=0)).float()
        assert torch.all(time_diffs >= 0), "Time differences must be non-negative."
        log_time_diffs, log_abs_det = log_and_log_abs_det(time_diffs)
        return ItemSpec(
            log_time_diffs,
            torch.tensor([MODALITY_CONTINUOUS] * log_time_diffs.shape[0]),
            log_abs_det,
            None,
            {}
        )


class NSNPPDataModule(BaseDataModule):
    def __init__(self, root: str, file_name: str, seed: int, **kwargs):
        path = Path(root) / (file_name % seed)
        print(path)
        if path.suffix == '.npy':
            data = np.load(path)
        elif path.suffix == '.pth':
            data = torch.load(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        super().__init__([max(len(series) + 1 for series in data)], 1)

        self.kwargs = kwargs

        self.data = data
        self.train = self.val = self.test = None

    def setup(self, stage: str):
        test_data_set_size = validation_data_set_size = min(10000, int(0.1 * len(self.data)))
        train_data_set_size = len(self.data) - test_data_set_size - validation_data_set_size

        self.train = NSPPPDataset(self.data[:train_data_set_size])
        self.val = NSPPPDataset(self.data[train_data_set_size:train_data_set_size + validation_data_set_size])
        self.test = NSPPPDataset(self.data[-test_data_set_size:])

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, collate_fn=batch_collate, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, collate_fn=batch_collate, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, collate_fn=batch_collate, **self.kwargs)
