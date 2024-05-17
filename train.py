import os
import lightning.pytorch as pl

from setup import config
from utils.util_model import LightningTripletNet
from utils.util_dataset import LightningDataModule


def main():

    dataset = LightningDataModule(config)
    triplet_net = LightningTripletNet(config)
    
    tqdm_cb = pl.callbacks.TQDMProgressBar()
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        filename="{epoch:02d}_",
        save_last=True
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    callbacks = [tqdm_cb, ckpt_cb, early_stopping_cb, lr_monitor_cb]

    trainer = pl.Trainer(accelerator="gpu",
                        devices=config.gpu_ids,
                        strategy="ddp",
                        max_epochs=config.total_epoch,
                        use_distributed_sampler=True,
                        precision="16-mixed",
                        callbacks=callbacks,
                        logger=True,
                        profiler="simple",
                        log_every_n_steps=1,
                        default_root_dir=config.base_dir)
    trainer.fit(triplet_net, dataset)

if __name__ == '__main__':
    main()
asda