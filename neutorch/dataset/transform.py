from abc import ABC, abstractmethod
import random

import numpy as np

from scipy.ndimage.filters import gaussian_filter

from .patch import Patch


DEFAULT_PROBABILITY = 0.5


class AbstractTransform(ABC):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        assert probability > 0.
        assert probability <= 1.
        self.probability = probability

    @property
    def name(self):
        return self.__class__.__name__

    def is_invertible(self):
        return hasattr(self, 'invert')

    def __call__(self, patch: Patch):
        if random.random() < self.probability:
            self.transform(patch)
        else:
            # for spatial transform, we need to correct the size
            # to make sure that the final patch size is correct 
            if hasattr(self, 'shrink_size'):
                patch.accumulate_delayed_shrink_size(self.shrink_size) 

    @abstractmethod
    def transform(self, patch: Patch):
        """perform the real transform of image and label

        Args:
            patch (Patch): image and label
        """
        pass

    
class SpatialTransform(AbstractTransform):
    """Modify image voxel position and reinterprete."""
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    @abstractmethod
    def transform(self, patch: Patch):
        """transform the image and label together

        Args:
            patch (tuple): image and label pair
        """
        pass
    
    @property
    @abstractmethod
    def shrink_size(self):
        """this transform might shrink the patch size.
        for example, droping a section will shrink the z axis.

        Return:
            shrink_size (tuple): z0,y0,x0,z1,y1,x1
        """
        return (0, 0, 0, 0, 0, 0)

class IntensityTransform(AbstractTransform):
    """change image intensity only"""
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    @abstractmethod
    def transform(self, patch: Patch):
        pass


class SectionTransform(AbstractTransform):
    """change a random section only."""
    def __init__(self, probability: float = DEFAULT_PROBABILITY ):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        self.selected_axis = random.randrange(3)
        self.selected_idx = random.randrange(
            patch.image.shape[self.selected_axis]
        )
        patch = self.transform_section(patch)
        return patch
    
    @abstractmethod
    def transform_section(self, patch: Patch):
        pass

class Compose(object):
    def __init__(self, transforms: list):
        """compose multiple transforms

        Args:
            transforms (list): list of transform instances
        """
        self.transforms = transforms
        shrink_size = np.zeros((6,), dtype=np.int64)
        for transform in transforms:
            if isinstance(transform, SpatialTransform):
                shrink_size += np.asarray(transform.shrink_size)
        self.shrink_size = tuple(x for x in shrink_size)

    def __call__(self, patch: Patch):
        for transform in self.transforms:
            transform(patch)



class DropSection(SpatialTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        # since this transform really removes information
        # we do not delay the shrinking
        # make the first and last section missing is meaning less
        b0, c0, z0, y0, x0 = patch.image.shape
        z = random.randrange(1, z0-1)
        image = np.zeros((b0, c0, z0-1, y0, x0), dtype=patch.image.dtype)
        label = np.zeros((b0, c0, z0-1, y0, x0), dtype=patch.label.dtype)
        image[..., :z, :, :] = patch.image[..., :z, :, :]
        label[..., :z, :, :] = patch.label[..., :z, :, :]
        image[..., z:, :, :] = patch.image[..., z+1, :, :]
        label[..., z:, :, :] = patch.label[..., z+1, :, :]
        patch.image = image
        patch.label = label

    @property
    def shrink_size(self):
        return (0, 0, 0, 1, 0, 0)


class BlackBox(IntensityTransform):
    def __init__(self,
            probability: float = DEFAULT_PROBABILITY,
            max_box_size: tuple = (4,4,4),
            max_box_num: int = 3):
        """make some black cubes in image patch

        Args:
            probability (float, optional): probability of triggering this augmentation. Defaults to 1..
            max_box_size (tuple, optional): maximum cube size. Defaults to (4,4,4).
            max_box_num (int, optional): maximum number of black boxes. Defaults to 2.
        """
        super().__init__(probability=probability)
        assert len(max_box_size) == 3
        self.max_box_size = max_box_size
        self.max_box_num = max_box_num

    def transform(self, patch: Patch):
        box_num = random.randint(1, self.max_box_num)
        for _ in range(box_num):
            box_size = tuple(random.randint(1, s) for s in self.max_box_size)
            start = tuple(random.randint(1, t-b) for t, b in zip(patch.shape[-3:], box_size))
            patch.image[
                ...,
                start[0] : start[0] + box_size[0],
                start[1] : start[1] + box_size[1],
                start[2] : start[2] + box_size[2],
            ] = 0

class NormalizeTo01(IntensityTransform):
    def __init__(self, probability: float = 1.):
        super().__init__(probability=probability)
        
    def transform(self, patch: Patch):
        if np.issubdtype(patch.image.dtype, np.uint8):
            patch.image = patch.image.astype(np.float32) / 255.

class AdjustBrightness(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            factor: float = 0.3):
        super().__init__(probability=probability)
        factor = np.clip(factor, 0, 2)
        self.factor = factor
    
    def transform(self, patch: Patch):
        patch.image += (random.random() - 0.5) * self.factor
        np.clip(patch.image, 0., 1., out=patch.image)

class AdjustContrast(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            factor: float = 0.3):
        super().__init__(probability=probability)
        factor = np.clip(factor, 0., 2.)
        self.factor = factor

    def transform(self, patch: Patch):
        factor = 1 + (random.random() - 0.5) * self.factor
        patch.image *= factor
        np.clip(patch.image, 0., 1., out=patch.image)


class Gamma(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        gamma = random.random() * 2. - 1.
        patch.image **= 2.** gamma


class GaussianBlur2D(IntensityTransform):
    def __init__(self, probability: float=DEFAULT_PROBABILITY, 
            sigma: float = 5.0):
        super().__init__(probability=probability)
        self.sigma = sigma

    def transform(self, patch: Patch):
        gaussian_filter(patch.image, sigma=self.sigma, output=patch.image)