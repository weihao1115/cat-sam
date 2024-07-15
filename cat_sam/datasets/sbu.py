import json
from os.path import join

from cat_sam.datasets.base import BinaryCATSAMDataset


few_shot_img_dict = {
    1: ['lssd1109'],
    16: [
        'lssd3356', 'lssd1409', 'lssd1090', 'lssd3732', 'lssd2011', 'lssd1901', 'lssd1821', 'lssd1512',
        'lssd3713', 'lssd1376', 'lssd3493', 'lssd3729', 'lssd610', 'lssd3008', 'lssd1319', 'lssd1109'
    ]
}


class SBUDataset(BinaryCATSAMDataset):

    def __init__(
            self,
            data_dir: str,
            train_flag: bool,
            shot_num: int = None,
            **super_args
    ):
        json_path = join(data_dir, 'train.json' if train_flag else 'test.json')
        with open(json_path, 'r') as j_f:
            json_config = json.load(j_f)
        for key in json_config.keys():
            json_config[key]['image_path'] = join(data_dir, json_config[key]['image_path'])
            json_config[key]['mask_path'] = join(data_dir, json_config[key]['mask_path'])

        if shot_num is not None:
            assert shot_num in [1, 16], f"Invalid shot_num: {shot_num}! Must be either 1 or 16!"
            json_config = {key: value for key, value in json_config.items() if key in few_shot_img_dict[shot_num]}

        super(SBUDataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=0, object_connectivity=8,
            area_threshold=10, relative_threshold=True, relative_threshold_ratio=0.001,
            **super_args
        )
