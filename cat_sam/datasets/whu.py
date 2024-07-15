import json
from os.path import join

from cat_sam.datasets.base import BinaryCATSAMDataset


few_shot_img_dict = {
    1: ['2432'],
    16: [
        '168', '515', '807', '1176', '1552', '1901', '2432', '2679',
        '3130', '3486', '3678', '3939', '4263', '4335', '4640', '4725'
    ]
}


class WHUDataset(BinaryCATSAMDataset):

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

        super(WHUDataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=254, object_connectivity=4,
            area_threshold=20, relative_threshold=False,
            ann_scale_factor=2, noisy_mask_threshold=0.0,
            **super_args
        )
