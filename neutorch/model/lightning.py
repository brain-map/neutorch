from functools import cached_property

import lightning.pytorch as pl
import torch
torch.set_float32_matmul_precision('medium')
from yacs.config import CfgNode

from .IsoRSUNet import Model
from neutorch.loss import BinomialCrossEntropyWithLogits


class LitIsoRSUNet(pl.LightningModule):
    def __init__(self, 
            cfg: CfgNode = None, 
            model: torch.nn.Module = None,
            learning_rate: float = 0.001) -> None:
        super().__init__()
        self.cfg = cfg

        self.learning_rate = learning_rate

        if model is None:
            model = Model(cfg.model.in_channels, cfg.model.out_channels)
        self.model = model
    
    def label_to_target(self, label: torch.Tensor):
        return label

    def post_processing(self, prediction: torch.Tensor):
        return prediction

    def forward(self, x):
        return self.model(x)
    
    @cached_property
    def loss_module(self):
        return BinomialCrossEntropyWithLogits()

    def training_step(self, batch, batch_idx: int):
        x, label = batch
        target = self.label_to_target(label)
        predict = self.model(x)
        predict = self.post_processing(predict)
        loss = self.loss_module(predict, target)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the test loop
        x, label = batch
        target = self.label_to_target(label)
        predict = self.model(x)
        test_loss = self.loss_module(predict, target)
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )