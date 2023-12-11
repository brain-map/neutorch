import os
import random
from abc import ABC, abstractproperty
from functools import cached_property
from time import time
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT

import numpy as np
from yacs.config import CfgNode
from chunkflow.lib.cartesian_coordinate import Cartesian

import lightning.pytorch as pl

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from neutorch.data.patch import collate_batch
from neutorch.loss import BinomialCrossEntropyWithLogits
from neutorch.model.io import load_chkpt, log_tensor, save_chkpt
from neutorch.model.IsoRSUNet import Model
from neutorch.data.dataset import worker_init_fn


class TrainerBase(pl.Trainer):
    def __init__(self, cfg: CfgNode, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(cfg, str) and os.path.exists(cfg):
            with open(cfg) as file:
                cfg = CfgNode.load_cfg(file)
        cfg.freeze()
               
        self.cfg = cfg
        self.patch_size=Cartesian.from_collection(cfg.train.patch_size)

    @cached_property
    def batch_size(self):
        # return self.num_gpus * self.cfg.train.batch_size
        # this batch size is for a single GPU rather than the total number!
        return self.cfg.train.batch_size

    # @cached_property
    # def path_list(self):
    #     glob_path = os.path.expanduser(self.cfg.dataset.glob_path)
    #     path_list = glob(glob_path, recursive=True)
    #     path_list = sorted(path_list)
    #     print(f'path_list \n: {path_list}')
    #     assert len(path_list) > 1
        
    #     # sometimes, the path list only contains the label without corresponding image!
    #     # assert len(path_list) % 2 == 0, \
    #         # "the image and synapses should be paired."
    #     return path_list

    # def _split_path_list(self):
    #     training_path_list = []
    #     validation_path_list = []
    #     for path in self.path_list:
    #         assignment_flag = False
    #         for validation_name in self.cfg.dataset.validation_names:
    #             if validation_name in path:
    #                 validation_path_list.append(path)
    #                 assignment_flag = True
            
    #         for test_name in self.cfg.dataset.test_names:
    #             if test_name in path:
    #                 assignment_flag = True

    #         if not assignment_flag:
    #             training_path_list.append(path)

    #     print(f'split {len(self.path_list)} ground truth samples to {len(training_path_list)} training samples, {len(validation_path_list)} validation samples, and {len(self.path_list)-len(training_path_list)-len(validation_path_list)} test samples.')
    #     self.training_path_list = training_path_list
    #     self.validation_path_list = validation_path_list

    @cached_property
    def model(self):
        model = Model(self.cfg.model.in_channels, self.cfg.model.out_channels)
                           
        if 'preload' in self.cfg.train:
            fname = self.cfg.train.preload
        else:
            fname = os.path.join(self.cfg.train.output_dir, 
                f'model_{self.cfg.train.iter_start}.chkpt')

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if os.path.exists(fname) and self.local_rank==0:
            model = load_chkpt(model, fname)

        return model

    @cached_property
    @abstractproperty
    def training_dataset(self):
        pass

    @cached_property
    def training_data_loader(self):
        
        dataloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    
    @cached_property
    def validation_data_loader(self):
        
        dataloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    @cached_property
    def validation_data_iter(self):
        validation_data_iter = iter(self.validation_data_loader)
        return validation_data_iter

    @cached_property
    def voxel_num(self):
        return np.product(self.patch_size) * self.batch_size

    def label_to_target(self, label: torch.Tensor):
        return label.cuda()

    def post_processing(self, prediction: torch.Tensor):
        if isinstance(self.loss_module, BinomialCrossEntropyWithLogits):
            return torch.sigmoid(prediction)
        else:
            return prediction
