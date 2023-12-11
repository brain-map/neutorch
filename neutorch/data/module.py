from functools import cached_property
from typing import List

import torch
import lightning as L

from chunkflow.lib.cartesian_coordinate import Cartesian
from neutorch.data.dataset import *


class Wasp(L.LightningDataModule):
    def __init__(self, 
            sample_config_files: List[str], 
            inputs: List[str] = ['image'],
            labels: List[str] = ['label'],
            output_patch_size: Cartesian = Cartesian(128, 128, 128),
            batch_size: int = 1,
            dataset_type: str = 'SemanticDataset',
            ) -> None:
        super().__init__()

        self.sample_config_files = sample_config_files
        self.inputs = inputs
        self.labels = labels
        self.output_patch_size = output_patch_size
        self.batch_size = batch_size
        self.dataset_type = eval(dataset_type)

        # # define the variable first
        # self.training_dataset = None
        # self.validation_dataset = None
        assert self.training_dataset is not None
        assert self.validation_dataset is not None
    
    @cached_property
    def training_dataset(self):
        return self.dataset_type.from_config_v5(
            self.sample_config_files, 
            mode='training', 
            inputs = self.inputs,
            labels = self.labels,
            output_patch_size=self.output_patch_size
        )

    @cached_property
    def validation_dataset(self):
        return self.dataset_type.from_config_v5(
            self.sample_config_files, 
            mode='validation',
            inputs = self.inputs,
            labels = self.labels,
            output_patch_size=self.output_patch_size,
        )

    def train_dataloader(self) :
        return torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            num_workers = 8,
            # prefetch_factor=2,
            drop_last=False,
            multiprocessing_context='spawn',
        )

    def val_dataloader(self) :
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers = 8,
            # prefetch_factor=2,
            drop_last=False,
            multiprocessing_context='spawn',
        )