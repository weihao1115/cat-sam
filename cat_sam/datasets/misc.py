import os
import random
from os.path import join, exists, relpath
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F


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


def find_random_points_in_objects(
        object_regions: List[np.ndarray],
        prompt_point_num: int = 1,
        random_num_prompt: bool = True
):
    # only select prompt points for the regions whose areas are larger than the threshold
    points = []
    for object_indices in object_regions:
        if random_num_prompt:
            # we randomly select a random number for each object during training
            _prompt_point_num = random.randint(1, prompt_point_num)
        else:
            _prompt_point_num = prompt_point_num

        # randomly select the given number of prompt points from each region
        random_idxs = np.random.permutation(object_indices.shape[0])[:_prompt_point_num]
        object_points = [[int(object_indices[idx][1]), int(object_indices[idx][0])] for idx in random_idxs]
        points.append(object_points)
    return points


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


def make_noisy_mask_on_objects(object_masks, scale_factor: int = 8, noisy_mask_threshold: float = 0.5):
    """
        Add noise to mask input
        From Mask Transfiner https://github.com/SysCV/transfiner
    """
    def get_incoherent_mask(input_masks):
        mask = input_masks.float()
        h, w = input_masks.shape[-2:]

        mask_small = F.interpolate(mask, (h // scale_factor, w // scale_factor), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (256, 256), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue


    noisy_object_masks = []
    for i, o_m in enumerate(object_masks):
        o_m_256 = F.interpolate(torch.from_numpy(o_m[None, None, :]), (256, 256), mode='bilinear')

        mask_noise = torch.randn(o_m_256.shape) * 1.0
        inc_masks = get_incoherent_mask(o_m_256)
        o_m_256 = ((o_m_256 + mask_noise * inc_masks) > noisy_mask_threshold).float()

        if len(o_m_256.shape) == 4 and o_m_256.size(0) == 1:
            o_m_256 = o_m_256.squeeze(0)
        noisy_object_masks.append(o_m_256)

    noisy_object_masks = torch.stack(noisy_object_masks, dim=0)
    return noisy_object_masks


def generate_prompts_from_mask(
        gt_mask: np.ndarray, tgt_prompts: List[str],
        object_connectivity: int = 8, area_threshold: int = 20,
        relative_threshold: bool = True, relative_threshold_ratio: float = 0.001,
        max_object_num: int = None, prompt_point_num: int = 1,
        ann_scale_factor: int = 8, noisy_mask_threshold: float = 0.5
):
    point_coords, box_coords, noisy_object_masks = None, None, None
    object_regions, object_masks = find_objects_from_mask(
        gt_mask, connectivity=object_connectivity,
        area_threshold=area_threshold, relative_threshold=relative_threshold,
        relative_threshold_ratio=relative_threshold_ratio
    )
    # skip prompt generation if no object is found in the gt_mask
    if object_regions is not None:
        if max_object_num is not None and len(object_regions) > max_object_num:
            random_object_idxs = np.random.permutation(len(object_regions))[:max_object_num]
            object_regions = [object_regions[idx] for idx in random_object_idxs]
            object_masks = np.stack([object_masks[idx] for idx in random_object_idxs], axis=0)

        if 'point' in tgt_prompts:
            point_coords = find_random_points_in_objects(
                object_regions, prompt_point_num=prompt_point_num
            )

        if 'box' in tgt_prompts:
            box_coords = find_bound_box_on_objects(object_regions)

        if 'mask' in tgt_prompts:
            noisy_object_masks = make_noisy_mask_on_objects(
                object_masks=object_masks, scale_factor=ann_scale_factor,
                noisy_mask_threshold=noisy_mask_threshold
            )
    # since object_masks act as the label for training, we give one zero mask when there is no object
    else:
        object_masks = np.zeros(shape=(1, gt_mask.shape[-2], gt_mask.shape[-1]), dtype=np.float32)

    return point_coords, box_coords, noisy_object_masks, object_masks
