import numpy as np
import torch
from torch.utils.data import DataLoader

from flex_tpp.data.base import FixedDiscretePatternDataset, BaseDataModule


def make_moons(n_samples=100, *, noise=None, seed=42):
    n_samples_out = n_samples_in = n_samples // 2

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if noise is not None:
        X += np.random.default_rng(seed).normal(size=X.shape) * noise

    return X, y


class TwoMoonsDataModule(BaseDataModule):
    def __init__(self, noise: float, **kwargs):
        super().__init__([1, 1], 1)
        assert noise == 0.1, "Noise is currently ignored"
        self.noise = noise
        self.kwargs = kwargs

        self.train = self.val = self.test = None

    def setup(self, stage: str):
        train_data_set_size = 10_000
        validation_data_set_size = 1_000
        test_data_set_size = 1_000
        data_set_size = train_data_set_size + validation_data_set_size + test_data_set_size
        noise = 0.1

        data = torch.from_numpy(
            make_moons(data_set_size, noise=noise)[0]
        ).float()[torch.randperm(data_set_size)]
        dim = data.shape[1]
        assert sum(self.dims) == dim

        discrete_pattern = torch.tensor([False, False])
        self.train = FixedDiscretePatternDataset(data[:train_data_set_size], discrete_pattern)
        self.val = FixedDiscretePatternDataset(data[train_data_set_size:train_data_set_size + validation_data_set_size], discrete_pattern)
        self.test = FixedDiscretePatternDataset(data[-test_data_set_size:], discrete_pattern)

    def train_dataloader(self):
        return DataLoader(self.train, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, **self.kwargs)


class GaussianMixtureDataModule(BaseDataModule):
    def __init__(self, num_modes: int, std: float = 0.5, class_distance: float = 2, class_first: bool = True, **kwargs):
        super().__init__([2], num_modes)
        self.kwargs = kwargs
        self.std = std
        self.class_distance = class_distance
        self.class_first = class_first

    def setup(self, stage: str):
        train_data_set_size = 10_000
        validation_data_set_size = 1_000
        test_data_set_size = 1_000
        data_set_size = train_data_set_size + validation_data_set_size + test_data_set_size

        class_label = torch.randint(low=0, high=self.max_num_classes, size=(data_set_size,), dtype=torch.long)
        position = torch.randn((data_set_size,)) * self.std + class_label * self.class_distance
        dim_order = 1 if self.class_first else -1
        data = torch.stack(
            [
                class_label.float(),
                position
            ][::dim_order],
            1
        )

        discrete_pattern = torch.tensor([True, False][::dim_order])
        self.train = FixedDiscretePatternDataset(data[:train_data_set_size], discrete_pattern)
        self.val = FixedDiscretePatternDataset(
            data[train_data_set_size:train_data_set_size + validation_data_set_size],
            discrete_pattern
        )
        self.test = FixedDiscretePatternDataset(data[-test_data_set_size:], discrete_pattern)

    def train_dataloader(self):
        return DataLoader(self.train, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, **self.kwargs)
