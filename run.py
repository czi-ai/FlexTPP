import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.get("print_and_exit", False):
        return
    data = instantiate(cfg.data)

    ## instantiate the model
    model = instantiate(
        cfg.model, dims=data.dims, max_num_classes=data.max_num_classes,
        optim_cfg=cfg.optim, _recursive_=False,
    )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer)
    logger = instantiate(trainer_kwargs.pop("logger", []))
    try:
        iter(logger)
    except TypeError:
        logger = [logger]
    for l in logger:
        l.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    fit_kwargs = trainer_kwargs.pop("fit_kwargs", {})
    callbacks = instantiate(trainer_kwargs.pop("callbacks", []))
    trainer = Trainer(**trainer_kwargs, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=data, **fit_kwargs)

    ckpt_callback = next(cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint))
    best_model_path = ckpt_callback.best_model_path
    ckpt = torch.load(best_model_path, weights_only=False)
    # Re-instantiate the model from checkpoint
    best_model = instantiate(
        cfg.model, dims=data.dims, max_num_classes=data.max_num_classes,
        optim_cfg=cfg.optim, _recursive_=False
    )
    best_model.load_state_dict(ckpt["state_dict"])
    # Validate and test the best model
    trainer.validate(model=best_model, datamodule=data)
    trainer.test(model=best_model, datamodule=data)
    return (
        trainer.callback_metrics.get("val/nll", torch.tensor(float("nan"))).item(),
    )


if __name__ == '__main__':
    main()
