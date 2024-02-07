import json
from os.path import curdir, join, exists
from cat_sam.datasets.misc import get_json_dict_from_dir


if __name__ == '__main__':
    test_data_dir = join(curdir, '3. The cropped image tiles and raster labels', 'val')
    train_data_dir = join(curdir, '3. The cropped image tiles and raster labels', 'train')
    assert exists(train_data_dir) and exists(test_data_dir), \
        "train or val folders don't exist! Please download them!"

    train_image_dir = join(train_data_dir, 'image')
    train_mask_dir = join(train_data_dir, 'label')
    train_json = get_json_dict_from_dir(
        image_dir=train_image_dir, mask_dir=train_mask_dir
    )
    train_json_path = join(curdir, 'train.json')
    with open(train_json_path, 'w') as f:
        json.dump(train_json, f, indent=4)


    test_image_dir = join(test_data_dir, 'image')
    test_mask_dir = join(test_data_dir, 'label')
    test_json = get_json_dict_from_dir(
        image_dir=test_image_dir, mask_dir=test_mask_dir
    )
    test_json_path = join(curdir, 'test.json')
    with open(test_json_path, 'w') as f:
        json.dump(test_json, f, indent=4)

