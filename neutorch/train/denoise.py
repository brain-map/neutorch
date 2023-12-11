#/bin/env python

from functools import cached_property

import click
from yacs.config import CfgNode
import torch

import lightning.pytorch as pl

from neutorch.data.dataset import VolumeWithMask
from ..model.lightning import LitIsoRSUNet

from neutorch.data.dataset import load_cfg

class DenoiseModel(LitIsoRSUNet):
    def __init__(self, cfg: CfgNode, model: torch.nn.Module = None) -> None:
        super().__init__(cfg, model)

    @cached_property
    def loss_module(self):
        return torch.nn.MSELoss() 

@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
@click.option('--devices', '-d', default="auto", 
    help='number of devices')
@click.option('--accelerator', '-a', type=click.Choice(['auto', 'gpu', 'cpu', 'cuda', 'hpu', 'ipu', 'mps', 'tpu']), default='auto',
    help='accelerator to use, [auto, cpu, cuda, hpu, ipu, mps, tpu]')
@click.option('--strategy', '-s', 
    type=click.Choice(['ddp', 'ddp_spawn', 'auto']), default='auto')
def main(config_file: str, devices: int, accelerator: str, strategy: str):
    cfg = load_cfg(config_file)

    training_dataset = VolumeWithMask.from_config(cfg, mode='training')
    validation_dataset = VolumeWithMask.from_config(cfg, mode='validation')
    
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=cfg.train.batch_size,
        num_workers = 8,
        # prefetch_factor=2,
        drop_last=False,
        multiprocessing_context='spawn',
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.train.batch_size,
        num_workers = 8,
        # prefetch_factor=2,
        drop_last=False,
        multiprocessing_context='spawn',
    )
    
    # torch.distributed.init_process_group(backend="nccl")
    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=devices, 
        strategy=strategy
    )

    trainer.fit(
        model = DenoiseModel(cfg), 
        train_dataloaders = training_dataloader,
        val_dataloaders = validation_dataloader,
    )



