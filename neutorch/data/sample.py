from abc import ABC, abstractmethod
import random
from typing import List, Union
from functools import cached_property

import numpy as np

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk import Chunk, load_chunk
from chunkflow.lib.synapses import Synapses

from neutorch.data.patch import Patch
from neutorch.data.transform import *

DEFAULT_PATCH_SIZE = Cartesian(128, 128, 128)
DEFAULT_NUM_CLASSES = 1


class AbstractSample(ABC):
    def __init__(self, output_patch_size: Cartesian):

        if isinstance(output_patch_size, int):
            output_patch_size = (output_patch_size,) * 3
        else:
            assert len(output_patch_size) == 3

        if not isinstance(output_patch_size, Cartesian):
            output_patch_size = Cartesian.from_collection(output_patch_size)
        self.output_patch_size = output_patch_size

    @property
    @abstractmethod
    def random_patch(self):
        pass

    @property
    def sampling_weight(self) -> int:
        """the weight to sample 

        Returns:
            int: the relative weight. The default is 1, 
                so all the sample have the same weight.
        """
        return 1 


class Sample(AbstractSample):
    def __init__(self, 
            images: List[Chunk], label: Union[np.ndarray, Chunk],
            output_patch_size: Cartesian, 
            forbbiden_distance_to_boundary: tuple = None) -> None:
        """Image sample with ground truth annotations

        Args:
            images (List[Chunk]): different versions of image chunks normalized to 0-1
            label (np.ndarray): training label
            patch_size (Cartesian): output patch size. this should be the patch_size before transform. 
                the patch is expected to be shrinked to be the output patch size.
            forbbiden_distance_to_boundary (Union[tuple, int]): 
                the distance from patch center to sample boundary that is not allowed to sample 
                the order is z,y,x,-z,-y,-x
                if this is an integer, then all dimension is the same.
                if this is a tuple of three integers, the positive and negative is the same
                if this is a tuple of six integers, the positive and negative 
                direction is defined separately. 
        """
        super().__init__(output_patch_size=output_patch_size)
        assert len(images) > 0
        assert images[0].ndim == 3
        assert label.ndim >= 3
        assert isinstance(label, Chunk)
        assert images[0].shape == label.shape[-3:], f'label voxel offset: {label.start}'
        
        if isinstance(label, Chunk):
            label = label.array

        # if images[0].shape != label.shape[-3:]:
        #     print(images[0].shape)
        #     print(label.shape[-3:])
        #     breakpoint()
        self.images = images
        self.label = label
        
        # print(f'output patch size: {self.output_patch_size}')
        assert isinstance(self.output_patch_size, Cartesian)
        for ps, ls in zip(self.output_patch_size, label.shape[-3:]):
            assert ls >= ps

        if forbbiden_distance_to_boundary is None:
            forbbiden_distance_to_boundary = self.patch_size_before_transform // 2 
        assert len(forbbiden_distance_to_boundary) == 3 or len(forbbiden_distance_to_boundary)==6

        for idx in range(3):
            # the center of random patch should not be too close to boundary
            # otherwise, the patch will go outside of the volume
            assert forbbiden_distance_to_boundary[idx] >= self.patch_size_before_transform[idx] // 2
            assert forbbiden_distance_to_boundary[-idx] >= self.patch_size_before_transform[-idx] // 2
        
        self.center_start = forbbiden_distance_to_boundary[:3]
        self.center_stop = tuple(s - d for s, d in zip(
            images[0].shape, forbbiden_distance_to_boundary[-3:]))
        for cs, cp in zip(self.center_start, self.center_stop):
            assert cp > cs

        # print(f'center start: {self.center_start}')
        # print(f'center stop: {self.center_stop}')
        # print(f'forbbiden distance to boundary: {forbbiden_distance_to_boundary}')
        # breakpoint()

    # @classmethod
    # def from_json(cls, json_file: str, patch_size: Cartesian = DEFAULT_PATCH_SIZE):
    #     with open(json_file, 'r') as jf:
    #         data = json.load(jf)
    #     return cls.from_dict(data, patch_size=patch_size)

    def _expand_to_5d(self, array: np.ndarray):
        if array.ndim == 3:
            return np.expand_dims(array, axis=(0, 1))
        elif array.ndim == 4:
            return np.expand_dims(array, axis=0)
        elif array.ndim == 5:
            return array
        else:
            raise ValueError('only support 3 to 5 dimensional array.')

    @property
    def random_patch(self):
        patch = self.random_patch_from_center_range(self.center_start, self.center_stop)
        
        # print(f'patch size before transform: {patch.shape}')
        self.transform(patch)
        # print(f'patch size after transform: {patch.shape}')
        # patch.apply_delayed_shrink_size()
        # print(f'patch size after shrink: {patch.shape}')
        assert patch.shape[-3:] == self.output_patch_size, \
            f'get patch shape: {patch.shape}, expected patch size {self.output_patch_size}'
        return patch
    
    def random_patch_from_center_range(self, center_start: tuple, center_stop: tuple):
        cz = random.randrange(center_start[0], center_stop[0])
        cy = random.randrange(center_start[1], center_stop[1])
        cx = random.randrange(center_start[2], center_stop[2])
        center = Cartesian(cz, cy, cx)
        return self.patch_from_center(center) 

    def patch_from_center(self, center: Cartesian):
        patch_size = self.patch_size_before_transform
        bz, by, bx = center - patch_size // 2
        # bz = center[0] - self.output_patch_size[-3] // 2
        # by = center[1] - self.output_patch_size[-2] // 2
        # bx = center[2] - self.output_patch_size[-1] // 2

        image = random.choice(self.images).array
        image_patch = image[...,
            bz : bz + patch_size[-3],
            by : by + patch_size[-2],
            bx : bx + patch_size[-1]
        ]
        label_patch = self.label[...,
            bz : bz + patch_size[-3],
            by : by + patch_size[-2],
            bx : bx + patch_size[-1]
        ]

        # print(f'start: {(bz, by, bx)}, patch size: {self.output_patch_size}')
        assert image_patch.shape[-1] == image_patch.shape[-2]
        if image_patch.shape[-3:] != self.patch_size_before_transform.tuple:
            breakpoint()
        assert image_patch.shape[-3:] == self.patch_size_before_transform.tuple, \
            f'image patch shape: {image_patch.shape}, patch size before transform: {self.patch_size_before_transform}'
        # if we do not copy here, the augmentation will change our 
        # image and label sample!
        image_patch = self._expand_to_5d(image_patch).copy()
        label_patch = self._expand_to_5d(label_patch).copy()
        return Patch(image_patch, label_patch)
    
    @cached_property
    def sampling_weight(self):
        weight = int(np.product(tuple(e-b for b, e in zip(
            self.center_start, self.center_stop))))
        
        # if len(np.unique(self.label)) == 1:
        #     # reduce the weight
        #     weight /= 10.

        return weight

    @cached_property
    def patch_size_before_transform(self):
        return self.output_patch_size + \
            self.transform.shrink_size[:3] + \
            self.transform.shrink_size[-3:]

    @cached_property
    def transform(self):
        return Compose([
            NormalizeTo01(probability=1.),
            AdjustContrast(),
            AdjustBrightness(),
            Gamma(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            MaskBox(),
            Perspective2D(),
            # RotateScale(probability=1.),
            DropSection(),
            Flip(),
            Transpose(),
            MissAlignment(),
        ])

 
    

class SampleWithPointAnnotation(Sample):
    def __init__(self, 
            images: List[Chunk], 
            annotation_points: np.ndarray,
            output_patch_size: Cartesian, 
            forbbiden_distance_to_boundary: tuple = None) -> None:
        """Image sample with ground truth annotations

        Args:
            image (np.ndarray): image normalized to 0-1
            annotation_points (np.ndarray): point annotations with zyx order.
            output_patch_size (Cartesian): output patch size
            forbbiden_distance_to_boundary (tuple, optional): sample patches far away 
                from sample boundary. Defaults to None.
        """

        assert annotation_points.shape[1] == 3
        self.annotation_points = annotation_points
        label = np.zeros_like(images[0].array, dtype=np.float32)
        label = self._points_to_label(label)
        super().__init__(
            images, label, 
            output_patch_size = output_patch_size,
            forbbiden_distance_to_boundary=forbbiden_distance_to_boundary
        )

    @property
    def sampling_weight(self):
        """use number of annotated points as weight to sample volume."""
        return int(self.annotation_points.shape[0])

    def _points_to_label(self, label: np.ndarray,
            expand_distance: int = 2) -> tuple:
        """transform point annotation to volumes

        Args:
            expand_distance (int): expand the point annotation to a cube. 
                This will help to got more positive voxels.
                The expansion should be small enough to ensure that all the voxels are inside T-bar.

        Returns:
            bin_presyn: binary label of annotated position.
        """
        # assert synapses['resolution'] == [8, 8, 8]
        # label = np.zeros_like(image, dtype=np.float32)
        # adjust label to 0.05-0.95 for better regularization
        # the effect might be similar with Focal loss!
        label += 0.05
        for idx in range(self.annotation_points.shape[0]):
            coordinate = self.annotation_points[idx, :]
            label[...,
                coordinate[0]-expand_distance : coordinate[0]+expand_distance,
                coordinate[1]-expand_distance : coordinate[1]+expand_distance,
                coordinate[2]-expand_distance : coordinate[2]+expand_distance,
            ] = 0.95
        assert np.any(label > 0.5)
        return label


class PostSynapseReference(AbstractSample):
    def __init__(self,
            synapses: Synapses,
            images: List[Chunk], 
            output_patch_size: Cartesian, 
            point_expand: int = 2,
        ):
        """Ground Truth for post synapses

        Args:
            synapses (Synapses): including both presynapses and postsynapses
            images (List[Chunk]): several image chunk versions covering the whole synapses
            patch_size (Cartesian): image patch size covering the whole synapse
            point_expand (int): expand the point. range from 1 to half of patch size.
        """
        super().__init__(output_patch_size=output_patch_size)

        self.images = images
        self.synapses = synapses
        self.pre_index2post_indices = synapses.pre_index2post_indices
        self.point_expand = point_expand

    @property
    def random_patch(self):
        pre_index = random.randrange(0, self.synapses.pre_num)
        pre = self.synapses.pre[pre_index, :]
        
        post_indices = self.pre_index2post_indices[pre_index]
        assert len(post_indices) > 0

        bbox = BoundingBox.from_center(
            Cartesian(*pre), 
            extent=self.output_patch_size // 2
        )

        image = random.choice(self.images)
        
        # Note that image is 4D array, the first dimension size is 1
        image = image.cutout(bbox)
        assert image.dtype == np.uint8
        image = image.astype(np.float32)
        image /= 255.
        # pre_label = np.zeros_like(image)
        # pre_label[
            
        #     pre[0] - self.point_expand : pre[0] + self.point_expand,
        #     pre[1] - self.point_expand : pre[1] + self.point_expand,
        #     pre[2] - self.point_expand : pre[2] + self.point_expand,
        # ] = 0.95

        # stack them together in the channel dimension
        # image = np.expand_dims(image, axis=0)
        # pre_label = np.expand_dims(pre_label, axis=0)
        # image = np.concatenate((image, pre_label), axis=0)

        label = np.zeros(image.shape, dtype=np.float32)
        label = Chunk(label, voxel_offset=image.voxel_offset)
        label += 0.05
        for post_index in post_indices:
            assert post_index < self.synapses.post_num
            coord = self.synapses.post_coordinates[post_index, :]
            coord = coord - label.voxel_offset
            label[...,
                coord[0] - self.point_expand : coord[0] + self.point_expand,
                coord[1] - self.point_expand : coord[1] + self.point_expand,
                coord[2] - self.point_expand : coord[2] + self.point_expand,
            ] = 0.95
        assert np.any(label > 0.5)

        return Patch(image, label)


class SemanticSample(Sample):
    def __init__(self, 
            images: List[Chunk], 
            label: Union[np.ndarray, Chunk], 
            output_patch_size: Cartesian, 
            num_classes: int = DEFAULT_NUM_CLASSES,
            forbbiden_distance_to_boundary: tuple = None) -> None:
        super().__init__(images, label, output_patch_size, forbbiden_distance_to_boundary)
        # number of classes
        self.num_classes = num_classes

    @classmethod
    def from_explicit_path(cls, 
            image_paths: list, label_path: str, 
            output_patch_size: Cartesian,
            num_classes: int=DEFAULT_NUM_CLASSES):
        label = load_chunk(label_path)
        print(f'label path: {label_path} with size {label.shape}')

        images = []
        for image_path in image_paths:
            image = load_chunk(image_path)
            images.append(image)
            print(f'image path: {image_path} with size {image.shape}')
        return cls(images, label, output_patch_size, num_classes=num_classes)

    @classmethod
    def from_label_path(cls, label_path: str, 
            output_patch_size: Cartesian,
            num_classes: int = DEFAULT_NUM_CLASSES):
        """construct a sample from a single file of label

        Args:
            label_path (str): the path of a label file

        Returns:
            an instance of a sample
        """
        image_path = label_path.replace('label', 'image')
        return cls.from_explicit_path(
            [image_path,], label_path, output_patch_size, num_classes=num_classes)

    @classmethod
    def from_explicit_dict(cls, d: dict, 
            output_patch_size: Cartesian,
            num_classes: int = DEFAULT_NUM_CLASSES):
        image_paths = d['images']
        label_path = d['label']
        return cls.from_explicit_path(
            image_paths, label_path, output_patch_size, num_classes=num_classes)

    @cached_property
    def voxel_num(self):
        return len(self.label)

    @cached_property
    def class_counts(self):
        return np.bincount(self.label.flatten(), minlength=self.num_classes)


class OrganelleSample(SemanticSample):
    def __init__(self, 
            images: List[Chunk], 
            label: Union[np.ndarray, Chunk], 
            output_patch_size: Cartesian, 
            num_classes: int = DEFAULT_NUM_CLASSES, 
            forbbiden_distance_to_boundary: tuple = None,
            skip_classes: list = None,
            selected_classes: list = None) -> None:
        super().__init__(images, label, output_patch_size, 
            num_classes=num_classes, 
            forbbiden_distance_to_boundary=forbbiden_distance_to_boundary)

        if skip_classes is not None:
            for class_idx in skip_classes:
                self.label.array[self.label.array>class_idx] -= 1
        
        if selected_classes is not None:
            self.label.array = np.isin(self.label.array, selected_classes)
    
    @cached_property
    def transform(self):
        return Compose([
            NormalizeTo01(probability=1.),
            # AdjustContrast(factor_range = (0.95, 1.8)),
            # AdjustBrightness(min_factor = 0.05, max_factor = 0.2),
            AdjustContrast(),
            AdjustBrightness(),
            Gamma(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            MaskBox(),
            # Perspective2D(),
            # RotateScale(probability=1.),
            DropSection(probability=1.),
            Flip(),
            Transpose(),
            # MissAlignment(),
        ])

class VolumeSample(AbstractSample):
    def __init__(self, output_patch_size: Cartesian):
        super().__init__(output_patch_size)

    @property
    def random_patch(self):
        return super().random_patch

    @property
    def sampling_weight(self) -> int:
        """None sampling weight will be replaced with average weight."""
        return None

if __name__ == '__main__':
    import os
    from tqdm import tqdm
    from PIL import Image
    from neutorch.data.dataset import load_cfg
    
    PATCH_NUM = 100
    DEFAULT_PATCH_SIZE=Cartesian(32, 64, 64)
    OUT_DIR = os.path.expanduser('~/dropbox/patches/')
    cfg = load_cfg('./config_mito.yaml')

    sample = SemanticSample.from_explicit_dict(
        cfg.dataset.validation.human, 
        output_patch_size=DEFAULT_PATCH_SIZE
    )
    
    for idx in tqdm(range(PATCH_NUM)):
        patch = sample.random_patch
        image = patch.image
        label = patch.label
        if image.shape[-3:] != DEFAULT_PATCH_SIZE.tuple:
            breakpoint()

        # section_idx = image.shape[-3]//2
        section_idx = 0
        image = image[0,0, section_idx, :,:]
        label = label[0,0, section_idx, :,:]

        image *= 255.
        im = Image.fromarray(image).convert('L')
        im.save(os.path.join(OUT_DIR, f'{idx}_image.jpg'))

        label *= 255
        lbl = Image.fromarray(label).convert('L')
        lbl.save(os.path.join(OUT_DIR, f'{idx}_label.jpg'))

    
