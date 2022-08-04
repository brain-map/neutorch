from typing import Union
from functools import cached_property

from chunkflow.lib.bounding_boxes import Cartesian

import torch

from neutorch.dataset.transform import *



class DatasetBase(torch.utils.data.IterableDataset):
    def __init__(self, 
            patch_size: Union[int, tuple, Cartesian]=(128, 128, 128),
        ):
        """
        Parameters:
            patch_size (int or tuple): the patch size we are going to provide.
        """
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = Cartesian(patch_size, patch_size, patch_size)
        else:
            patch_size = Cartesian.from_collection(patch_size)

        assert isinstance(self.transform, Compose)

        self.patch_size = patch_size

        self.patch_size_before_transform = self.patch_size + \
            self.transform.shrink_size[:3] + \
            self.transform.shrink_size[-3:]

        # inherite this class and build the samples
        self.samples = None

    @cached_property
    def sample_num(self):
        return len(self.samples)

    def _compute_sample_weights(self):
        # use the number of candidate patches as volume sampling weight
        sample_weights = []
        for sample in self.samples:
            sample_weights.append(sample.sampling_weight)

        self.sample_weights = sample_weights
    
    def __next__(self):
        # only sample one subject, so replacement option could be ignored
        sample_index = random.choices(
            range(0, self.sample_num),
            weights=self.sample_weights,
            k=1,
        )[0]
        sample = self.samples[sample_index]
        patch = sample.random_patch
        self.transform(patch)
        patch.apply_delayed_shrink_size()
        # print('patch shape: ', patch.shape)
        assert patch.shape[-3:] == self.patch_size, \
            f'get patch shape: {patch.shape}, expected patch size {self.patch_size}'
        
        patch.to_tensor()
        return patch.image, patch.target

    def __iter__(self):
        """generate random patches from samples

        Yields:
            tuple[tensor, tensor]: image and target tensors
        """
        while True:
            yield next(self)

    @cached_property
    def transform(self):
        return Compose([
            NormalizeTo01(probability=1.),
            AdjustBrightness(),
            AdjustContrast(),
            Gamma(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            BlackBox(),
            Perspective2D(),
            # RotateScale(probability=1.),
            #DropSection(),
            Flip(),
            Transpose(),
            MissAlignment(),
        ])
