# Kvasir-SEG Dataset

Please follow the commands below to prepare the Kvasir-SEG dataset:
```
$: pwd
/your_dir/cat-sam
$: cd ./data/kvasir
$: gdown https://drive.google.com/uc?id=1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb
$: gdown https://drive.google.com/uc?id=1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao
$: unzip TrainDataset.zip
$: unzip TestDataset.zip
$: rm TrainDataset.zip TestDataset.zip
$: conda activate cat_sam
(cat-sam) $: ls
preprocess.py  README.md  TestDataset  TrainDataset
(cat-sam) $: python preprocess.py
(cat-sam) $: ls
preprocess.py  README.md  TestDataset  test.json  TrainDataset  train.json
```

If you cannot download the dataset by `gdown`, 
please download the data in the 2nd step of the section 3.1 in the [PraNet](https://github.com/DengPingFan/PraNet?tab=readme-ov-file#31-trainingtesting).

