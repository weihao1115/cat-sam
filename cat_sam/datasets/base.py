import random
from typing import Dict, List, Union

import cv2
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from cat_sam.datasets.transforms import Compose

from cat_sam.datasets.misc import generate_prompts_from_mask


class BaseSegDataset(Dataset):

    def __init__(
            self,
            dataset_config: Union[Dict, List[Dict]],
            label_threshold: Union[int, None] = 128,
            transforms: List = None
    ):
        self.label_threshold = label_threshold
        self.transforms = Compose(transforms) if transforms else None

        if isinstance(dataset_config, Dict):
            dataset_config_list = [dataset_config]
        elif isinstance(dataset_config, List):
            dataset_config_list = dataset_config
        else:
            raise RuntimeError(
                    f"Your given dataset_config should be either a dict or a list of dicts, "
                    f"but got {type(dataset_config)}!"
            )
        self.idx2img_gt_path = {}
        for config_dict in dataset_config_list:
            self.idx2img_gt_path.update(config_dict)
        self.idx_list = list(self.idx2img_gt_path.keys())


    def __len__(self):
        return len(self.idx_list)


    def __getitem__(self, index):
        index_name = self.idx_list[index]
        image = Image.open(self.idx2img_gt_path[index_name]['image_path']).convert('RGB')
        gt_mask = Image.open(self.idx2img_gt_path[index_name]['mask_path']).convert('L')
        image, gt_mask = np.array(image, dtype=np.float32), np.array(gt_mask, dtype=np.float32)
        # For the 0-255 mask, discretize its values into 0.0 or 1.0
        if self.label_threshold is not None:
            gt_mask = np.where(gt_mask > self.label_threshold, 1.0, 0.0)

        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], repeats=3, axis=-1)
        elif len(image.shape) == 3 and image.shape[0] == 1:
            image = np.repeat(image, repeats=3, axis=-1)
        elif len(image.shape) != 3 and image.shape[0] != 3:
            raise RuntimeError(f'Wrong image shape: {image.shape}. It should be either [H, W] or [H, W, 1] or [H, W, 3]!')

        if len(gt_mask.shape) == 3 and gt_mask.shape[0] in [1, 3]:
            gt_mask = gt_mask[:, :, 0]
        elif len(gt_mask.shape) != 2:
            raise RuntimeError(f'Wrong mask shape: {gt_mask.shape}. It should be either [H, W] or [H, W, 1] or [H, W, 3]!')

        if image.shape[:2] != gt_mask.shape[:2]:
            if image.shape[0] == gt_mask.shape[1] and image.shape[1] == gt_mask.shape[0]:
                image = np.transpose(image, (1, 0, 2))
            else:
                image = cv2.resize(image, (gt_mask.shape[1], gt_mask.shape[0]))

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=gt_mask)
            image, gt_mask = transformed["image"], transformed['mask']

        return dict(
            images=image, gt_masks=gt_mask, index_name=index_name
        )


    @classmethod
    def collate_fn(cls, batch):
        # preprocess List[Dict[str, Any]] to Dict[str, List[Any]]
        batch_dict = dict()
        while len(batch) != 0:
            ele_dict = batch[0]
            if ele_dict is not None:
                for key in ele_dict.keys():
                    if key not in batch_dict.keys():
                        batch_dict[key] = []
                    batch_dict[key].append(ele_dict[key])
            # remove the redundant data for memory safety
            batch.remove(ele_dict)

        batch_dict['images'] = [torch.from_numpy(item).permute(2, 0, 1) for item in batch_dict['images']]
        batch_dict['gt_masks'] = [torch.from_numpy(item) for item in batch_dict['gt_masks']]
        batch_dict['object_masks'] = \
            [torch.from_numpy(item) if item is not None else None for item in batch_dict['object_masks']]

        # pad the prompt points with placeholders (-1) to make sure that each prompt has the same number of points
        point_coords, point_labels = [], []
        for item in batch_dict['point_coords']:
            # give a None value to the images without any prompt points
            if item is None:
                point_coords.append(None)
                point_labels.append(None)
            # all the labels of prompt points are either foreground points (label=1) or placeholder (label=-1)
            else:
                _point_coords, _point_labels = item, []
                max_num_coords = max(len(_p_c) for _p_c in _point_coords)
                for _p_c in _point_coords:
                    _point_labels.append([1 for _ in _p_c])

                    curr_num_coords = len(_p_c)
                    if curr_num_coords < max_num_coords:
                        _p_c.extend([[0, 0] for _ in range(max_num_coords - curr_num_coords)])
                        _point_labels[-1].extend([-1 for _ in range(max_num_coords - curr_num_coords)])

                point_coords.append(torch.FloatTensor(_point_coords))
                point_labels.append(torch.LongTensor(_point_labels))
        batch_dict['point_coords'] = point_coords
        batch_dict['point_labels'] = point_labels

        # give a None value to the images without any prompt boxes
        batch_dict['box_coords'] = \
            [torch.FloatTensor(item) if item is not None else None for item in batch_dict['box_coords']]

        # we don't stack all the image and ground-truth tensors together to deal with different image sizes
        return batch_dict



class BinaryCATSAMDataset(BaseSegDataset):

    def __init__(
            self,
            train_flag: bool,
            offline_prompt_points: Union[str, List[str]] = None,
            prompt_point_num: int = 1,
            max_object_num: int = None,
            object_connectivity: int = 8,
            area_threshold: int = 20,
            relative_threshold: bool = True,
            relative_threshold_ratio: float = 0.001,
            ann_scale_factor: int = 8,
            noisy_mask_threshold: float = 0.5,
            **super_args
    ):
        super(BinaryCATSAMDataset, self).__init__(**super_args)
        self.train_flag = train_flag
        self.prompt_kwargs = dict(
            object_connectivity=object_connectivity,
            area_threshold=area_threshold,
            relative_threshold=relative_threshold,
            relative_threshold_ratio=relative_threshold_ratio,
            max_object_num=max_object_num,
            prompt_point_num=prompt_point_num,
            ann_scale_factor=ann_scale_factor,
            noisy_mask_threshold=noisy_mask_threshold
        )

        self.offline_prompt_points = None
        if offline_prompt_points is not None:
            self.offline_prompt_points = {}

            if isinstance(offline_prompt_points, str):
                offline_prompt_points = [offline_prompt_points]
            for item in offline_prompt_points:
                self.offline_prompt_points.update(item)


    def __getitem__(self, index):
        ret_dict = super(BinaryCATSAMDataset, self).__getitem__(index)
        point_coords, box_coords, noisy_object_masks, object_masks = generate_prompts_from_mask(
            gt_mask=ret_dict['gt_masks'],
            tgt_prompts=[random.choice(['point', 'box', 'mask'])] if self.train_flag else ['point', 'box'],
            **self.prompt_kwargs
        )
        # offline random prompt points for evaluation
        if self.offline_prompt_points is not None:
            point_coords = self.offline_prompt_points[ret_dict['index_name']]

        ret_dict.update(
            point_coords=point_coords,
            box_coords=box_coords,
            noisy_object_masks=noisy_object_masks,
            object_masks=object_masks
        )
        return ret_dict
