import yaml

import pytorch_lightning as pl
import click

from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_module import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = LightningModule(config)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**config['model_checkpoint'])

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger("lightning_logs", name=config["experiment_name"], default_hp_metric=False)

    trainer = pl.Trainer(
        logger=logger,
        gpus=config['gpus'],
        max_epochs=config['epochs'],
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()