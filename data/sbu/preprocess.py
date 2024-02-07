import json
from os.path import curdir, join, exists
from cat_sam.datasets.misc import get_json_dict_from_dir


if __name__ == '__main__':
    test_data_dir = join(curdir, 'SBU-shadow', 'SBU-Test')
    train_data_dir = join(curdir, 'SBU-shadow', 'SBUTrain4KRecoveredSmall')
    assert exists(train_data_dir) and exists(test_data_dir), \
        "SBUTrain4KRecoveredSmall and SBU-Test don't exist! Please download them from Google Driver!"

    train_image_dir = join(train_data_dir, 'ShadowImages')
    train_mask_dir = join(train_data_dir, 'ShadowMasks')
    train_json = get_json_dict_from_dir(
        image_dir=train_image_dir, mask_dir=train_mask_dir, mask_ext='png'
    )
    train_json_path = join(curdir, 'train.json')
    with open(train_json_path, 'w') as f:
        json.dump(train_json, f, indent=4)


    test_image_dir = join(test_data_dir, 'ShadowImages')
    test_mask_dir = join(test_data_dir, 'ShadowMasks')
    test_json = get_json_dict_from_dir(
        image_dir=test_image_dir, mask_dir=test_mask_dir, mask_ext='png'
    )
    test_json_path = join(curdir, 'test.json')
    with open(test_json_path, 'w') as f:
        json.dump(test_json, f, indent=4)

