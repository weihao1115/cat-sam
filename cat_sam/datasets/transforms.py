import math
import random
import torch.nn.functional as F
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch


class BaseTransform(ABC):
    def __init__(self, p: float = 1.0, **kwargs):
        assert 0.0 < p <= 1.0
        self.p = p
        self.end_init_hook(**kwargs)

    def end_init_hook(self, **kwargs):
        pass

    def __call__(self, img: np.ndarray, mask: np.ndarray = None):
        if random.random() < self.p:
            return self.apply(img, mask)
        else:
            return img, mask

    @abstractmethod
    def apply(self, image: np.ndarray, mask: np.ndarray = None):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: List[BaseTransform]):
        if isinstance(transforms, BaseTransform):
            transforms = [transforms]
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        for t in self.transforms:
            res = t(image, mask)
            if not isinstance(res, Tuple):
                image = res
            elif len(res) == 2:
                image, mask = res
            else:
                raise RuntimeError
        return dict(image=image, mask=mask)


class VerticalFlip(BaseTransform):
    def apply(self, image: np.ndarray, mask: np.ndarray = None):
        image = np.ascontiguousarray(image[::-1, ...])
        if mask is not None:
            mask = np.ascontiguousarray(mask[::-1, ...])
        return image, mask


class HorizontalFlip(BaseTransform):
    def apply(self, image: np.ndarray, mask: np.ndarray = None):
        image = np.ascontiguousarray(image[:, ::-1, ...])
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, ::-1, ...])
        return image, mask


class RandomCrop(BaseTransform):
    def end_init_hook(self, scale: Union[List[float], float] = [0.1, 1.0]):
        if isinstance(scale, float):
            scale = [scale, scale]
        # assert scale[0] > 0.0 and 0.0 < scale[1] <= 1.0
        assert scale[0] > 0.0
        self.scale = scale


    def apply(self, image: np.ndarray, mask: np.ndarray = None):
        height, width = image.shape[:2]
        area = height * width
        aspect_ratio = width / height
        while True:
            crop_factor = random.uniform(*self.scale)
            target_area = crop_factor * area
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            if 0 < crop_height <= height and 0 < crop_width <= width:
                break

        height_crop_start = random.randint(0, height - crop_height)
        height_crop_end = height_crop_start + crop_height
        width_crop_start = random.randint(0, width - crop_width)
        width_crop_end = width_crop_start + crop_width

        image = image[height_crop_start:height_crop_end, width_crop_start:width_crop_end]
        if mask is not None:
            mask = mask[height_crop_start:height_crop_end, width_crop_start:width_crop_end]
        return image, mask
