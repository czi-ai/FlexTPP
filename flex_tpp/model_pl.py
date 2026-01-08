import torch
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

from flex_tpp.data.base import BatchSpec, MODALITY_CONTINUOUS
from flex_tpp.model import FlexTPPModel, PropertyMTPPFlexTPPModel


def mean_metrics(metrics, prefix=""):
    return {
        f"{prefix}{key}": value.mean(-1) if value.shape != () else value
        for key, value in metrics.items()
    }


class FlexTPPModule(pl.LightningModule):
    CLASS = FlexTPPModel

    def __init__(self, *, dims, dim_k, n_head, optim_cfg, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.CLASS(
            dim=sum(dims),
            dim_m=dim_k * n_head,
            dim_k=dim_k,
            **kwargs
        )
        self.optim_cfg = optim_cfg

    def compute_metrics(self, batch: BatchSpec, compute_dataset_metrics=False):
        ll = self.model.log_prob(batch)

        metrics = {
            "nll": -ll[torch.isfinite(batch.data)].mean(),
        }
        disc_select = torch.isfinite(batch.data) & (batch.types != MODALITY_CONTINUOUS)
        if disc_select.any():
            metrics["nll_disc"] = -ll[disc_select].mean()
        cont_select = torch.isfinite(batch.data) & (batch.types == MODALITY_CONTINUOUS)
        if cont_select.any():
            metrics["nll_cont"] = -ll[cont_select].mean()

        if compute_dataset_metrics:
            metrics.update(self.trainer.datamodule.val_dataset.validation_metrics(batch, self.model))
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = mean_metrics(self.compute_metrics(batch), "train/")
        self.log_dict(metrics, prog_bar=True)
        return metrics["train/nll"]

    def validation_step(self, batch, batch_idx):
        metrics = mean_metrics(self.compute_metrics(batch, compute_dataset_metrics=False), "val/")
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        metrics = mean_metrics(self.compute_metrics(batch, compute_dataset_metrics=False), "test/")
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        config = OmegaConf.to_container(self.optim_cfg)
        config["optimizer"] = instantiate(config["optimizer"], self.parameters())
        if "lr_scheduler" in config:
            assert "scheduler" in config["lr_scheduler"]
            if config["lr_scheduler"]["scheduler"]["_target_"] == "torch.optim.lr_scheduler.OneCycleLR":
                config["lr_scheduler"]["scheduler"] = instantiate(
                    config["lr_scheduler"]["scheduler"],
                    config["optimizer"],
                    config["optimizer"].param_groups[0]["lr"],
                )
            else:
                config["lr_scheduler"]["scheduler"] = instantiate(
                    config["lr_scheduler"]["scheduler"], config["optimizer"]
                )
        return config


class PropertyMTPPFlexTPPModule(FlexTPPModule):
    CLASS = PropertyMTPPFlexTPPModel
