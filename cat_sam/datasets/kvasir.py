import json
from os.path import join

from cat_sam.datasets.base import BinaryCATSAMDataset



class KvasirDataset(BinaryCATSAMDataset):

    def __init__(
            self,
            data_dir: str,
            train_flag: bool,
            **super_args
    ):
        json_path = join(data_dir, 'train.json' if train_flag else 'test.json')
        with open(json_path, 'r') as j_f:
            json_config = json.load(j_f)
        for key in json_config.keys():
            json_config[key]['image_path'] = join(data_dir, json_config[key]['image_path'])
            json_config[key]['mask_path'] = join(data_dir, json_config[key]['mask_path'])

        super(KvasirDataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=254, object_connectivity=8,
            area_threshold=20, relative_threshold=True,
            **super_args
        )
