import hydra
import os
import torch
import pytorch_lightning as pl
from models.vaub_pnp_DA import VAUB_pnp_DA

def save_backbone(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, "model.pth")
    torch.save(model.state_dict(), model_path)

@hydra.main(config_path="config", config_name="config_lenet")
def run(cfg):
    pl.seed_everything(cfg.seed)
    if cfg.logger:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project=f"{cfg.name}",
            name=f"{cfg.dataset.src}2{cfg.dataset.tgt}"
        )
    else:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=f"{cfg.logger_path}/{cfg.name}",
            name=f"{cfg.dataset.src}2{cfg.dataset.tgt}")
        logger.log_hyperparams(cfg)

    model = VAUB_pnp_DA(cfg)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        devices=cfg.gpus,
        logger=logger,
        max_epochs=cfg.epoch,
        accelerator='gpu',
        strategy='ddp',
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(model)
    trainer.test()

    if cfg.save:
        path = os.path.join(cfg.save_path, f"{cfg.name}")
        save_backbone(model, path)

if __name__ == '__main__':
    run()