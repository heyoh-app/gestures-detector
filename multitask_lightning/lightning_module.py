import json
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.dataset import Dataset
from losses.loss import get_loss
from models.model import UnetClipped
from utils.train_utils import get_optim
from metrics.metric import MeanAveragePrecision


class LightningModule(pl.LightningModule):

    def __init__(self, config):
        super(LightningModule, self).__init__()
        self.hparams.update(config)

        self.net = UnetClipped(**self.hparams["model"])

        self.image_size = self.hparams["train_data_params"]["size"]
        self.stride = self.hparams["train_data_params"]["output_stride"]
        self.in_channels = self.hparams["model"]["in_channels"]
        self.num_classes = sum(self.hparams["train_data_params"]["subclasses"])

        with open(self.hparams["split"]) as json_file:
            split = json.load(json_file)

        self.train_files = split["train"]
        self.val_files = split["val"]

        self.train_data = Dataset(self.train_files, **self.hparams['train_data_params'])
        self.val_data = Dataset(self.val_files, **self.hparams['val_data_params'])

        self.loss_functions = {task: get_loss(params) for task, params in self.hparams['losses'].items()}
        self.metric = MeanAveragePrecision(
            num_classes=self.num_classes,
            out_img_size=self.hparams["val_data_params"]["size"] // self.hparams["val_data_params"]["output_stride"],
            **self.hparams["metric"]
        )

        self.freeze_epochs = self.hparams["freeze_epochs"]
        if self.freeze_epochs != 0:
            self.set_grad(False)

    def set_grad(self, requires_grad: bool):
        for param in self.net.encoder.features.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        return self.net(x)

    def _log(self, log_dict):
        [self.log(log_name, log_value, on_step=False, on_epoch=True, prog_bar=True) for log_name, log_value in
         log_dict.items()]

    def compute_losses(self, masks_predict, masks_gt, mode: str = "train"):
        masks, predicts, losses = dict(), dict(), dict()

        masks['kpoint'], masks['side'], masks['size'] = masks_gt
        predicts['kpoint'], predicts['side'], predicts['size'] = torch.split(masks_predict, [self.num_classes, 1, 1], 1)

        loss = torch.tensor(0.).cuda()
        for task in masks.keys():
            losses[task] = self.loss_functions[task](predicts[task], masks[task])
            self.log(f"{mode}_loss_{task}", losses[task], on_step=False, on_epoch=True, prog_bar=True)
            loss += losses[task]
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        image = batch[0]
        output = self.forward(image)
        loss = self.compute_losses(output, batch[1:], mode="train")
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        image = batch[0]
        output = self.forward(image)
        loss = self.compute_losses(output, batch[1:], mode="val")
        self.metric.update(output, batch[1:])
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss = np.mean([output['val_loss'].detach().cpu() for output in outputs])
        val_map = self.metric.pascal_map_value(reset=True)
        self.log('val_map', val_map, on_step=False, on_epoch=True, prog_bar=True)

        if self.current_epoch + 1 == self.freeze_epochs and self.freeze_epochs != 0:
            self.set_grad(True)   # unfreeze encoder
        return {'val_loss': val_loss, 'val_map': val_map}

    def configure_optimizers(self):
        optimizer = get_optim(self.hparams["optimizer"])
        optimizer = optimizer(self.net.parameters())

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.hparams['reduce_on_plateau']),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': self.hparams["check_val_every_n_epoch"]
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams["workers"]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams["val_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams["workers"]
        )
