from typing import Union

import torch
from lightning.pytorch.callbacks import Callback, TQDMProgressBar
import lightning.pytorch as pl


def shorten_key(key: str, chars_to_keep: int) -> str:
    parts = key.split("/")
    return "/".join([p[:chars_to_keep] for p in parts])


class ConciseProgressBar(TQDMProgressBar):
    def __init__(self, *args, min_length: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_length = min_length

    def get_metrics(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> dict[str, Union[int, str, float, dict[str, float]]]:
        metrics = super().get_metrics(trainer, pl_module)
        if len(metrics) == 0:
            return metrics
        for chars_to_keep in range(self.min_length, max(len(key) for key in metrics) - 1):
            shortened_metrics = {
                shorten_key(key, chars_to_keep): value
                for key, value in metrics.items()
            }
            if len(shortened_metrics) == len(metrics):
                return shortened_metrics
        return metrics


class BestCheckpointReloadLRDecayCallback(Callback):
    def __init__(self, monitor='val_loss', mode='min', patience=3, decay_factor=0.1, min_lr=1e-6):
        """
        Args:
            monitor (str): the metric name to monitor (e.g. 'val_loss').
            mode (str): "min" if lower values are better, "max" if higher values are better.
            patience (int): number of consecutive epochs with no improvement before reloading and decaying.
            decay_factor (float): factor to multiply the learning rate by.
            min_lr (float): minimum learning rate.
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_score = None

    def on_validation_end(self, trainer, pl_module):
        # Get the current metric from the logged metrics.
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)
        if current is None:
            # If the monitored metric is not available, do nothing.
            return

        # On the first validation, initialize best_score.
        if self.best_score is None:
            self.best_score = current
            self.wait = 0
            return

        # Check for improvement.
        improved = False
        if self.mode == 'min' and current < self.best_score:
            improved = True
        elif self.mode == 'max' and current > self.best_score:
            improved = True

        if improved:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1

        # If no improvement for patience epochs, reload best checkpoint and decay LR.
        all_param_groups = [
            param_group 
            for optimizer in trainer.optimizers 
            for param_group in optimizer.param_groups
        ]
        all_groups_above_min_lr = all(param_group['lr'] > self.min_lr for param_group in all_param_groups)
        if self.wait >= self.patience and all_groups_above_min_lr:
            # Access the ModelCheckpoint callback to get the best checkpoint path.
            best_ckpt_path = trainer.checkpoint_callback.best_model_path
            if best_ckpt_path:
                print(f"Reloading best checkpoint: {best_ckpt_path} and reducing LR by factor {self.decay_factor}")
                # Load the best checkpoint's state_dict.
                checkpoint = torch.load(best_ckpt_path, map_location=pl_module.device)
                pl_module.load_state_dict(checkpoint['state_dict'])
                # Adjust the learning rate for each optimizer.
                for optimizer in trainer.optimizers:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * self.decay_factor, self.min_lr)
            self.wait = 0

        for logger in trainer.loggers:
            logger.log_metrics({"lr_wait": self.wait}, step=trainer.global_step)
