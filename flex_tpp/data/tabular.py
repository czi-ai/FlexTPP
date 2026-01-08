from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader



class TabularDataModule(LightningDataModule):
    def __init__(self, table_name, data_dir: str = "data", **kwargs):
        super().__init__()
        self.table_name = table_name
        self.data_dir = Path(data_dir)
        self.kwargs = kwargs

        self.train = self.val = self.test = None
        if table_name == "gas":
            self.dims = [8]
        elif table_name == "power":
            self.dims = [6]
        elif table_name == "hepmass":
            self.dims = [21]

    def setup(self, stage: str):
        if self.table_name == "gas":
            from flex_tpp.data.gas import GAS
            # Obtain the -converted file by loading with pandas<2.0 and dumping it with .to_pickle
            gas = GAS(self.data_dir / "gas/ethylene_CO-converted.pickle")

            self.train = gas.trn
            self.val = gas.val
            self.test = gas.tst
        elif self.table_name == "power":
            from flex_tpp.data.power import POWER
            power = POWER(self.data_dir / "power/data.npy")
            self.train = power.trn
            self.val = power.val
            self.test = power.tst
        elif self.table_name == "hepmass":
            from flex_tpp.data.hepmass import HEPMASS
            hepmass = HEPMASS(self.data_dir / "hepmass")
            self.train = hepmass.trn
            self.val = hepmass.val
            self.test = hepmass.tst
        assert self.dims == [self.train[0].shape[0]]

    def train_dataloader(self):
        return DataLoader(self.train, **self.kwargs)

    def val_dataloader(self):
        kwargs = self.kwargs.copy()
        if "shuffle" in kwargs:
            kwargs.pop("shuffle")
        return DataLoader(self.val, **kwargs)

    def test_dataloader(self):
        kwargs = self.kwargs.copy()
        if "shuffle" in kwargs:
            kwargs.pop("shuffle")
        return DataLoader(self.test, **kwargs)
