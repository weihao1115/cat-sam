import os
from os.path import join, exists, relpath
from typing import List

import cv2
import numpy as np


def get_json_dict_from_dir(
        image_dir: str, mask_dir: str, mask_ext: str = None
):
    json_dict = {}
    for image_name in os.listdir(image_dir):
        sample_name = '.'.join(image_name.split('.')[:-1])

        image_file_path = join(image_dir, image_name)
        if not exists(image_file_path):
            raise RuntimeError(f'{image_file_path} does not exist! Please check!')

        mask_name = image_name
        if mask_ext:
            mask_name = '.'.join([sample_name, mask_ext])
        mask_file_path = join(mask_dir, mask_name)
        if not exists(mask_file_path):
            raise RuntimeError(f'{mask_file_path} does not exist! Please check!')

        # relative path in the current directory will be stored for compatibility
        json_dict[sample_name] = dict(
            image_path=relpath(image_file_path), mask_path=relpath(mask_file_path)
        )
    return json_dict


def find_objects_from_mask(
        mask: np.ndarray,
        connectivity: int = 8,
        area_threshold: int = 20,
        relative_threshold: bool = True,
        relative_threshold_ratio: float = 0.001,
):
    # from https://github.com/KyanChen/RSPrompter/blob/cky/tools/ins_seg/dataset_converters/whu_building_convert.py
    # Here, we only consider the mask values 1.0 as positive class, i.e., 255 pixel values
    object_num, objects_im, stats, centroids = cv2.connectedComponentsWithStats(
        image=mask.astype(np.uint8), connectivity=connectivity)

    # if no foreground object is found, a tuple of None is returned
    if object_num < 2:
        return None, None

    object_areas, object_indices_all, object_masks = [], [], []
    for i in range(1, object_num):
        object_mask = (objects_im == i).astype(np.float32)
        object_masks.append(object_mask)

        object_indices = np.argwhere(object_mask)
        object_areas.append(len(object_indices))
        object_indices_all.append(object_indices)

    # update area_threshold if relative_threshold is set
    if relative_threshold and len(object_indices_all) > 0:
        max_area = max(object_areas)
        area_threshold = max(max_area * relative_threshold_ratio, area_threshold)

    valid_objects = [i for i, o_a in enumerate(object_areas) if o_a > area_threshold]
    # if no foreground object is valid (area larger than the threshold), a tuple of None is returned
    if len(valid_objects) == 0:
        return None, None

    object_indices_all = [object_indices_all[i] for i in valid_objects]
    object_masks = [object_masks[i] for i in valid_objects]
    return object_indices_all, np.stack(object_masks, axis=0)



def find_bound_box_on_objects(object_regions: List[np.ndarray]):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if len(object_regions) == 0:
        return None

    boxes = []
    for object_indices in object_regions:
        y_max, x_max = object_indices[:, 0].max(), object_indices[:, 1].max()
        y_min, x_min = object_indices[:, 0].min(), object_indices[:, 1].min()
        # for each object, we append a 2-dim List for representing its bound box
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes



def generate_prompts_from_mask(
        gt_mask: np.ndarray, object_connectivity: int = 8, area_threshold: int = 20,
        relative_threshold: bool = True, relative_threshold_ratio: float = 0.001,
):
    box_coords = None
    object_regions, object_masks = find_objects_from_mask(
        gt_mask, connectivity=object_connectivity,
        area_threshold=area_threshold, relative_threshold=relative_threshold,
        relative_threshold_ratio=relative_threshold_ratio
    )
    # skip prompt generation if no object is found in the gt_mask
    if object_regions is not None:
        box_coords = find_bound_box_on_objects(object_regions)
    # since object_masks act as the label for training, we give one zero mask when there is no object
    else:
        object_masks = np.zeros(shape=(1, gt_mask.shape[-2], gt_mask.shape[-1]), dtype=np.float32)

    return box_coords, object_masks
