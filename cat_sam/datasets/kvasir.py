import json
from os.path import join

from cat_sam.datasets.base import BinaryCATSAMDataset


few_shot_img_dict = {
    1: ['cju0roawvklrq0799vmjorwfv'],
    16: [
        'cju7dn24o296i09871qfxb8s2',
        'cju83mki1jv5w0817kubxm31r',
        'cju6vucxvvlda0755j7msqnya',
        'cju43kj2pm34f0850l28ahpni',
        'cju2x7vw87mu30878hye2ca0m',
        'cju5ddda9bkkt0850enzwatb1',
        'cju77vvcwzcm50850lzoykuva',
        'cju3128yi0rpu0988o4oo5n8n',
        'cju8b7aqtr4a00987coba14b7',
        'cjz14qsk2wci60794un9ozwmw',
        'cju0s690hkp960855tjuaqvv0',
        'cju888fr7nveu0818r9uwtiit',
        'cju8b542nr81x0871uxnkm9ih',
        'cju2qtee81yd708787bsjr75d',
        'cju83h9ysjwe808716nt35oah',
        'cju0roawvklrq0799vmjorwfv'
    ]
}


class KvasirDataset(BinaryCATSAMDataset):

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

        super(KvasirDataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=254, object_connectivity=8,
            area_threshold=20, relative_threshold=True,
            **super_args
        )
