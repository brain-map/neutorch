import random
from abc import ABC, abstractmethod
from functools import cached_property

# import cv2
import numpy as np
from chunkflow.lib.cartesian_coordinate import Cartesian
from scipy.ndimage.filters import gaussian_filter
# from skimage.transform import swirl
from skimage.util import random_noise

from .patch import Patch

try:
    from reneu.lib.segmentation import seg_to_affs, remove_contact_xy
except ImportError:
    pass
# from copy import deepcopy


# from scipy.ndimage import affine_transform

DEFAULT_PROBABILITY = .5
DEFAULT_SHRINK_SIZE = (0, 0, 0, 0, 0, 0)
# DEFAULT_SHRINK_SIZE = None

# consider inherite from torchvision.transforms?
class AbstractTransform(ABC):
    def __init__(self, 
            probability: float = DEFAULT_PROBABILITY,
            validation: bool = False):
        """transform the image and label

        Args:
            probability (float, optional): probability to use this transform. Defaults to DEFAULT_PROBABILITY.
            validation (bool, optional): use it in the validation mode or not. Defaults to False.
        """
        assert probability > 0.
        assert probability <= 1.
        self.probability = probability
        self.valiation = validation
    
    def __str__(self) -> str:
        return 'AbstractTransform'

    @property
    def name(self):
        return self.__class__.__name__

    def is_invertible(self):
        return hasattr(self, 'invert')

    def __call__(self, patch: Patch):
        if random.random() < self.probability:
            patch = self.transform(patch)
            # for spatial transform, we need to correct the size
            # to make sure that the final patch size is correct
            self.shrink(patch)
        else:
            patch.shrink(self.shrink_size)
        # return patch
        
    @abstractmethod
    def transform(self, patch: Patch):
        """perform the real transform of image and label

        Args:
            patch (Patch): image and label
        """ 
        return patch

    @cached_property
    def shrink_size(self):
        """this transform might shrink the patch size.
        for example, droping a section will shrink the z axis.

        Return:
            shrink_size (tuple): z0,y0,x0,z1,y1,x1
        """
        return DEFAULT_SHRINK_SIZE
    
    def shrink(self, patch):
        if self.shrink_size != DEFAULT_SHRINK_SIZE:
            patch.shrink(self.shrink_size)
        # return patch

    
class SpatialTransform(AbstractTransform):
    """Modify image voxel position and reinterprete."""
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            validation: bool = False):
        super().__init__(probability=probability, validation=validation)

    def __str__(self) -> str:
        return 'SpatialTransform'

    @abstractmethod
    def transform(self, patch: Patch):
        """transform the image and label together

        Args:
            patch (tuple): image and label pair
        """
        return patch
    
class IntensityTransform(AbstractTransform):
    """change image intensity only"""
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def __str__(self) -> str:
        return 'IntensityTransform'

    @abstractmethod
    def transform(self, patch: Patch):
        return patch        


class SectionTransform(AbstractTransform):
    """change a random section only."""
    def __init__(self, probability: float = DEFAULT_PROBABILITY ):
        super().__init__(probability=probability)

    def __str__(self) -> str:
        return 'SectionTransform'

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


class OneOf(AbstractTransform):
    def __init__(self, transforms: list, 
            probability: float = DEFAULT_PROBABILITY) -> None:
        super().__init__(probability=probability)
        assert len(transforms) > 1
        self.transforms = transforms

    def __str__(self) -> str:
        return 'OneOf'

    @cached_property
    def shrink_size(self):
        shrink_size = np.zeros((6,), dtype=np.int64)
        for transform in self.transforms:
            if isinstance(transform, SpatialTransform):
                shrink_size += np.asarray(transform.shrink_size)
        return tuple(x for x in shrink_size)

    def transform(self, patch: Patch):
        # select one of the transforms
        transform_class = random.choice(self.transforms)
        patch = transform_class(patch)
        return patch


class DropSection(SpatialTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        z = random.randrange(1, patch.shape[-3])
        patch.image.array[..., z:-1, :, :] = patch.image[..., z+1:, :, :]
        patch.image.array = patch.image.array[..., :-1, :, :]
        
        patch.label.array[..., z:-1, :, :] = patch.label[..., z+1:, :, :]
        patch.label.array = patch.label.array[..., :-1, :, :]
        
        if patch.has_mask:
            patch.mask.array[..., z:-1, :, :] = patch.mask[..., z+1:, :, :]
            patch.mask.array = patch.mask.array[..., :-1, :, :]
        return patch

    @cached_property
    def shrink_size(self):
        return (0, 0, 0, 1, 0, 0)
    
    def shrink(self, patch):
        # the section is already dropped and the patch is shrinked
        pass

class MaskBox(IntensityTransform):
    def __init__(self,
            probability: float = DEFAULT_PROBABILITY,
            max_box_size: Cartesian = Cartesian(16,16,16),
            max_density: float = 0.1,
            max_box_num: int = None):
        """make some black cubes in image patch

        Args:
            probability (float, optional): probability of triggering this augmentation. Defaults to 1..
            max_box_size (tuple, optional): maximum cube size. Defaults to (4,4,4).
            max_box_num (int, optional): maximum number of black boxes. Defaults to 2.
        """
        super().__init__(probability=probability)
        assert len(max_box_size) == 3

        if not isinstance(max_box_size, Cartesian):
            max_box_size = Cartesian.from_collection(max_box_size)

        self.max_box_size = max_box_size
        self.max_box_num = max_box_num
        self.max_density = max_density

    def __str__(self) -> str:
        return 'MaskBox'

    def box_num(self, patch_size: tuple):
        if self.max_box_num is None:
            shape = Cartesian.from_collection(patch_size[-3:])
            self.max_box_num = int(np.prod(shape * self.max_density))
        
        return random.randint(1, self.max_box_num)
        
    def transform(self, patch: Patch):
        for _ in range(self.box_num(patch.shape)):
            box_size = tuple(random.randint(1, s) for s in self.max_box_size)
            # randint is inclusive
            start = tuple(random.randrange(1, t-b) for t, b in zip(patch.shape[-3:], box_size))
            if random.random() > 0.5:
                box = np.random.rand(*box_size)
            else:
                box = np.ones(box_size, dtype = patch.image.dtype) * random.random()
            patch.image[
                ...,
                start[0] : start[0] + box_size[0],
                start[1] : start[1] + box_size[1],
                start[2] : start[2] + box_size[2],
            ] = box
        return patch

class NormalizeTo01(IntensityTransform):
    def __init__(self, probability: float = 1., 
            normalize_label: bool=False):
        super().__init__(probability=probability)
        self.normalize_label = normalize_label
    
    def __str__(self) -> str:
        return 'NormalizeTo01'

    def transform(self, patch: Patch):
        if np.issubdtype(patch.image.dtype, np.uint8):
            patch.image.array = patch.image.array.astype(np.float32)
            patch.image.array = patch.image.array / 255.

        if self.normalize_label and np.issubdtype(
                patch.label.dtype, np.uint8) :
            patch.label.array = patch.label.array.astype(np.float32)
            patch.label.array = patch.label.array / 255.

        return patch

class AdjustBrightness(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            min_factor: float = 0.05,
            max_factor: float = 0.2):
        super().__init__(probability=probability)
        max_factor = np.clip(max_factor, 0, 2)
        self.min_factor = min_factor
        self.max_factor = max_factor
    
    def transform(self, patch: Patch):
        brightness = random.uniform(-0.5, 0.5) * random.uniform(
            self.min_factor, self.max_factor)
        if patch.image.array.mean() + brightness < 0.9:
            patch.image.array += brightness
            np.clip(patch.image.array, 0., 1., out=patch.image.array)
        return patch

class AdjustContrast(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            factor_range: tuple = (0.2, 2.)):
        super().__init__(probability=probability)
        factor_range = np.clip(factor_range, 0., 2.)
        self.factor_range = factor_range

    def __str__(self) -> str:
        return 'AdjustContrast'

    def transform(self, patch: Patch):
        #factor = 1 + random.uniform(-0.5, 0.5) * random.uniform(
        #    self.factor_range[0], self.factor_range[1])
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        if np.mean(patch.image) * factor < 0.9:
            patch.image *= factor
            np.clip(patch.image.array, 0., 1., out=patch.image.array)
        return patch


class Gamma(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def __str__(self) -> str:
        return 'Gamma'

    def transform(self, patch: Patch):
        # gamma = random.random() * 2. - 1.
        gamma = random.uniform(-1., 1.)
        patch.image **= 2.** gamma
        return patch

class GaussianBlur2D(IntensityTransform):
    def __init__(self, probability: float=DEFAULT_PROBABILITY, 
            sigma: float = 1.5):
        super().__init__(probability=probability)
        self.sigma = sigma

    def __str__(self) -> str:
        return 'GaussianBlur2D'

    def transform(self, patch: Patch):
        sigma = random.uniform(0.2, self.sigma)
        gaussian_filter(patch.image.array, sigma=sigma, output=patch.image.array)
        return patch


class GaussianBlur3D(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            max_sigma: tuple = (1.5, 1.5, 1.5)):
        super().__init__(probability=probability)
        self.max_sigma = max_sigma

    def __str__(self) -> str:
        return 'GaussianBlur3D'

    def transform(self, patch: Patch):
        sigma = tuple(random.uniform(0.2, s) for s in self.max_sigma)
        gaussian_filter(patch.image.array, sigma=sigma, output=patch.image.array)
        return patch


class Noise(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            mode: str='gaussian', max_variance: float = 0.02):
        super().__init__(probability=probability)
        self.mode = mode  
        self.max_variance = max_variance

    def __str__(self) -> str:
        return 'Noise'

    def transform(self, patch: Patch):
        variance = random.uniform(0.01, self.max_variance)
        random_noise(patch.image.array, mode=self.mode, var=variance)
        np.clip(patch.image.array, 0., 1., out=patch.image.array)
        return patch


class Flip(SpatialTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def __str__(self) -> str:
        return 'Flip'

    def transform(self, patch: Patch):
        axis_num = random.randint(1, 3)
        axis4d = random.sample(range(1,4), axis_num)

        # the image and label is 4d
        # the first axis is channel
        # the batch channel will be handled by the pytorch lightning
        patch.image.array = np.flip(patch.image.array, axis=axis4d)
        patch.label.array = np.flip(patch.label.array, axis=axis4d)
        if patch.has_mask:
            patch.mask.array = np.flip(patch.mask.array, axis=axis4d)

        # shrink = list(patch.delayed_shrink_size)
        # for ax in axis:
        #     # swap the axis to be shrinked
        #     shrink[3+ax], shrink[ax] = shrink[ax], shrink[3+ax]
        # patch.delayed_shrink_size = tuple(shrink)
        return patch


class Transpose(SpatialTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def __str__(self) -> str:
        return 'Transpose'

    def transform(self, patch: Patch):
        # only transform at XY
        axis = [2,3]
        random.shuffle(axis)
        if axis == [3,2]:
            axis4d = (0, 1, *axis,)
            patch.image.array = np.transpose(patch.image.array, axis4d)
            patch.label.array = np.transpose(patch.label.array, axis4d)
            if patch.has_mask:
                patch.mask.array = np.transpose(patch.mask.array, axis4d)
        return patch

    
class MissAlignment(SpatialTransform):
    def __init__(self, probability: float=DEFAULT_PROBABILITY,
            max_displacement: int=2):
        """move part of volume alone x axis
        We'll alwasy select a position alone z axis, and move the bottom part alone X axis.
        By combining with transpose, flip and rotation, we can get other displacement automatically.

        Args:
            probability (float, optional): probability of this augmentation. Defaults to DEFAULT_PROBABILITY.
            max_displacement (int, optional): maximum displacement. Defaults to 2.
        """
        super().__init__(probability=probability)
        assert max_displacement > 0
        assert max_displacement < 8
        self.max_displacement = max_displacement

    def __str__(self) -> str:
        return 'MissAlignment'

    def transform(self, patch: Patch):
        axis = random.randint(2, 4)
        displacement = random.randint(1, self.max_displacement)
        # random direction
        # no need to use random direction because we can combine with rotation and flipping
        # displacement *= random.choice([-1, 1])
        sz, sy, sx = patch.shape[-3:]
        if axis == 2:
            # displacement at Z
            zloc = random.randrange(1, sz)
            patch.image.array[..., zloc:, 
                self.max_displacement : sy-self.max_displacement,
                self.max_displacement : sx-self.max_displacement,
                ] = patch.image[..., zloc:, 
                    self.max_displacement+displacement : sy+displacement-self.max_displacement,
                    self.max_displacement+displacement : sx+displacement-self.max_displacement,
                    ]
            patch.label.array[..., zloc:, 
                self.max_displacement : sy-self.max_displacement,
                self.max_displacement : sx-self.max_displacement,
                ] = patch.label[..., zloc:, 
                    self.max_displacement+displacement : sy+displacement-self.max_displacement,
                    self.max_displacement+displacement : sx+displacement-self.max_displacement,
                    ]
            if patch.has_mask:
                patch.mask.array[..., zloc:, 
                    self.max_displacement : sy-self.max_displacement,
                    self.max_displacement : sx-self.max_displacement,
                    ] = patch.label[..., zloc:, 
                        self.max_displacement+displacement : sy+displacement-self.max_displacement,
                        self.max_displacement+displacement : sx+displacement-self.max_displacement,
                        ]

        elif axis == 3:
            # displacement at Y
            yloc = random.randrange(1, sy)
            
            # print('right side shape: ', patch.image[...,  self.max_displacement+displacement : sz+displacement-self.max_displacement,yloc:,self.max_displacement+displacement : sx+displacement-self.max_displacement,].shape)

            patch.image.array[..., 
                self.max_displacement : sz-self.max_displacement,
                yloc:, 
                self.max_displacement : sx-self.max_displacement,
                ] = patch.image[..., 
                    self.max_displacement+displacement : sz+displacement-self.max_displacement, 
                    yloc:, 
                    self.max_displacement+displacement : sx+displacement-self.max_displacement,
                    ]
            patch.label.array[..., 
                self.max_displacement : sz-self.max_displacement,
                yloc:,
                self.max_displacement : sx-self.max_displacement,
                ] = patch.label[..., 
                    self.max_displacement+displacement : sz+displacement-self.max_displacement,
                    yloc:, 
                    self.max_displacement+displacement : sx+displacement-self.max_displacement,
                    ]
            if patch.has_mask:
                patch.mask.array[..., 
                    self.max_displacement : sz-self.max_displacement,
                    yloc:,
                    self.max_displacement : sx-self.max_displacement,
                    ] = patch.mask[..., 
                        self.max_displacement+displacement : sz+displacement-self.max_displacement,
                        yloc:, 
                        self.max_displacement+displacement : sx+displacement-self.max_displacement,
                        ]

        elif axis == 4:
            # displacement at X
            xloc = random.randint(1, sx-1)
            patch.image.array[..., 
                self.max_displacement : sz-self.max_displacement,
                self.max_displacement : sy-self.max_displacement,
                xloc:, 
                ] = patch.image[...,  
                    self.max_displacement+displacement : sz+displacement-self.max_displacement,
                    self.max_displacement+displacement : sy+displacement-self.max_displacement,
                    xloc:
                    ]
            patch.label.array[..., 
                self.max_displacement : sz-self.max_displacement,
                self.max_displacement : sy-self.max_displacement,
                xloc:,
                ] = patch.label[..., 
                    self.max_displacement+displacement : sz+displacement-self.max_displacement,
                    self.max_displacement+displacement : sy+displacement-self.max_displacement,
                    xloc:, 
                    ]

            if patch.has_mask:
                patch.mask.array[..., 
                    self.max_displacement : sz-self.max_displacement,
                    self.max_displacement : sy-self.max_displacement,
                    xloc:,
                    ] = patch.mask[..., 
                        self.max_displacement+displacement : sz+displacement-self.max_displacement,
                        self.max_displacement+displacement : sy+displacement-self.max_displacement,
                        xloc:, 
                        ]

        return patch

    @cached_property
    def shrink_size(self):
        # return (0, 0, 0, 0, 0, self.max_displacement)
        return (self.max_displacement,) * 6

    def shrink(self, patch: Patch):
        patch.shrink(self.shrink_size)

# class Perspective2D(SpatialTransform):
    # def __init__(self, probability: float=DEFAULT_PROBABILITY,
            # corner_ratio: float=0.2):
        # """Warp image using Perspective transform
# 
        # Args:
            # probability (float, optional): probability of this transformation. Defaults to DEFAULT_PROBABILITY.
            # corner_ratio (float, optional): We split the 2D image to four equal size rectangles.
                # For each axis in rectangle, we further divid it to four rectangles using this ratio.
                # The rectangle containing the image corner was used as a sampling point region.
                # This idea is inspired by this example:
                # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#perspective-transformation
                # Defaults to 0.5.
        # """
        # super().__init__(probability=probability)
        # self.corner_ratio = corner_ratio
# 
    # def __str__(self) -> str:
        # return 'Perspective2D'
# 
    # def transformation_matrix(self, sy: int, sx: int):
        # corner_ratio = random.uniform(0.02, self.corner_ratio)
        # corner_ratio = self.corner_ratio
        # upper_left_point = [
            # random.randint(0, round(sy*corner_ratio/2)), 
            # random.randint(0, round(sx*corner_ratio/2))
        # ]
        # upper_right_point = [
            # random.randint(0, round(sy*corner_ratio/2)),
            # random.randint(sx-round(sx*corner_ratio/2), sx-1)
        # ]
        # lower_left_point = [
            # random.randint(sy-round(sy*corner_ratio/2), sy-1),
            # random.randint(0, round(sx*corner_ratio/2))
        # ]
        # lower_right_point = [
            # random.randint(sy-round(sy*corner_ratio/2), sy-1),
            # random.randint(sx-round(sx*corner_ratio/2), sx-1)
        # ]
        # pts1 = [
            # upper_left_point, 
            # upper_right_point, 
            # lower_left_point, 
            # lower_right_point
        # ]
        # push the list order to get rotation effect
        # for example, push one position will rotate about 90 degrees
#         push_index = random.randint(0, 3)
#         if push_index > 0:
#             tmp = deepcopy(pts1)
#             pts1[push_index:] = tmp[:4-push_index]
#             # the pushed out elements should be reversed
#             pts1[:push_index] = tmp[4-push_index:][::-1]
# # 
        # pts1 = np.asarray(pts1, dtype=np.float32)
        # 
        # pts2 =np.float32([[0, 0], [0, sx], [sy, 0], [sy, sx]])
        # M = cv2.getPerspectiveTransform(pts1, pts2)
        # return M
# 
    # def transform(self, patch: Patch):
        # matrix = np.eye(3)
        # offset = tuple(-ps // 2 for ps in patch.shape[-3:] )
        # sy, sx = patch.shape[-2:]
        # M = self.transformation_matrix(sy, sx)
        # for batch in range(patch.shape[0]):
            # for channel in range(patch.shape[1]):
                # for z in range(patch.shape[2]):
                    # patch.image[batch,channel,z,...] = self._transform2d(
                        # patch.image[batch, channel, z, ...], cv2.INTER_LINEAR, M, sy, sx
                    # )
                    # patch.label[batch,channel,z,...] = self._transform2d(
                        # patch.label[batch, channel, z, ...], cv2.INTER_NEAREST, M, sy, sx
                    # )
        # return patch
                # 
    # def _transform2d(self, arr: np.ndarray, interpolation: int, M: np.ndarray, sy: int, sx: int):
        # dst = cv2.warpPerspective(arr, M, (sy, sx), flags=interpolation)
        # return dst


# class RotateScale(SpatialTransform):
#     def __init__(self, probability: float=DEFAULT_PROBABILITY,
#             max_scaling: float=1.3):
#         super().__init__(probability=probability)
#         raise NotImplementedError('this augmentation is not working correctly yet. The image and label could have patchy effect.We are not sure why.')
#         self.max_scaling = max_scaling

#     def transform(self, patch: Patch):
#         # because we do not know the rotation angle
#         # we should apply the shrinking first
#         patch.apply_delayed_shrink_size()

#         # if the rotation is close to diagnal, for example 45 degree
#         # the label could be outside the volume and be black!
#         # angle = random.choice([0, 90, 180, -90, -180]) + random.randint(-5, 5)
#         angle = random.randint(0, 180)
#         scale = random.uniform(1.1, self.max_scaling)
#         center = patch.center[-2:]
#         mat = cv2.getRotationMatrix2D( center, angle, scale )
        
#         for batch in range(patch.shape[0]):
#             for channel in range(patch.shape[1]):
#                 for z in range(patch.shape[2]):
#                     patch.image[batch, channel, z, ...] = cv2.warpAffine(
#                         patch.image[batch, channel, z, ...],
#                         mat, patch.shape[-2:], flags=cv2.INTER_LINEAR
#                     ) 
#                     patch.label[batch, channel, z, ...] = cv2.warpAffine(
#                         patch.label[batch, channel, z, ...],
#                         mat, patch.shape[-2:], flags=cv2.INTER_NEAREST
#                     ) 


#class Swirl(SpatialTransform):
#    def __init__(self, max_rotation: int = 5, max_strength: int = 3, probability: float = DEFAULT_PROBABILITY):
#        super().__init__(probability=probability)
#        self.max_strength = max_strength
#        self.max_rotation = max_rotation
#
#    def __str__(self) -> str:
#        return 'Swirl'
#
#    def transform(self, patch: Patch):
#        """todo: transform label as well!
#
#        Args:
#            patch (Patch): _description_
#
#        Returns:
#            _type_: _description_
#        """
#        for z in range(patch.shape[-3]):
#            patch.image[..., z, :, :] = swirl(
#                patch.image[..., z, :, :],
#                rotation=random.randint(1, self.max_rotation),
#                strength=random.randint(1, self.max_strength),
#                radius = (patch.shape[-1] + patch.shape[-2]) // 4,
#            )
#        return patch

class Label2AffinityMap(SpatialTransform):
    def __init__(self, probability: float = 1., 
            validation: bool=True):
        """If this transform is used, the probability should always be 1.0."""
        assert probability == 1.
        super().__init__(probability, validation=validation)

    def __str__(self) -> str:
        return 'Label2AffinityMap'

    def transform(self, patch: Patch):
        """transform the label to affinity map."""
        assert patch.label.shape[0] == 1
        assert patch.label.shape[1] == 1
        assert patch.label.ndim == 4
        seg = patch.label.array[0,0,...]
        seg = seg.astype(np.uint64)
        remove_contact_xy(seg)
        seg = seg.astype(np.uint64)
        affs_ref = seg_to_affs(seg)
        patch.label.array =  np.expand_dims(affs_ref, axis=0)
        patch.label.voxel_offset += Cartesian(1,1,1)
        # print(f'patch shape after Label2AffinityMap: {patch.shape}')
        return patch

    @cached_property
    def shrink_size(self):
        return (1, 1, 1, 0, 0, 0)
    
    def shrink(self, patch: Patch):
        # the label is already shrinked, so we only need to shrink the image
        patch.image.shrink(self.shrink_size)
        if patch.has_mask:
            patch.mask.shrink(self.shrink_size)

class Compose(object):
    def __init__(self, transforms: list):
        """compose multiple transforms

        Args:
            transforms (list): list of transform instances
        """
        self.transforms = transforms

    def __str__(self) -> str:
        return '-->'.join([str(x) for x in self.transforms])

    @cached_property
    def shrink_size(self):
        shrink_size = np.zeros((6,), dtype=np.int64)
        for transform in self.transforms:
            if isinstance(transform, SpatialTransform):
                shrink_size += np.asarray(transform.shrink_size)
        return tuple(x for x in shrink_size)

    def __call__(self, patch: Patch):
        for transform in self.transforms:
            # print(f'patch size before {transform} with shrink size of {transform.shrink_size}: {patch.shape}')
            transform(patch)
            # print(f'patch size after {transform} with shrink size of {transform.shrink_size}: {patch.shape}')
        # after the transformation, the stride of array
        # could be negative, and pytorch could not tranform
        # the array to Tensor. Copy can fix it.
        # print(f'patch shape after Compose call: {patch.shape}')
        patch.image.array = patch.image.array.copy()
        patch.label.array = patch.label.array.copy()
        if patch.has_mask:
            patch.mask.array = patch.mask.array.copy()

