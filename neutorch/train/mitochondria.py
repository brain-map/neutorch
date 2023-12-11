#/bin/env python


import click

import lightning.pytorch as pl

from neutorch.model.lightning import LitIsoRSUNet
from neutorch.model.IsoRSUNet import Model
from neutorch.data.module import Wasp as WaspDataModule

from neutorch.data.dataset import load_cfg


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
@click.option('--chkpt-file', '-k',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True),
    default= None,
    help = 'trained model checkpoint file'
)
def main(config_file: str, devices: int, accelerator: str, strategy: str, chkpt_file: str):
    cfg = load_cfg(config_file)

    datamodule = WaspDataModule(
        cfg.samples, 
        inputs = cfg.inputs,
        labels = cfg.labels,
        output_patch_size = cfg.train.patch_size,
        batch_size = cfg.train.batch_size,
        dataset_type=cfg.dataset_type,
    )

    # torch.distributed.init_process_group(backend="nccl")
    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=devices, 
        strategy=strategy
    )

    
    if chkpt_file is not None:
        model = LitIsoRSUNet.load_from_checkpoint(
            chkpt_file,
            model=Model(cfg.model.in_channels, cfg.model.out_channels),
        )
    else:
        model = LitIsoRSUNet(cfg)

    trainer.fit(
        model = model, 
        datamodule=datamodule,
    )



