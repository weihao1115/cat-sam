# SBU-Shadow Dataset

Please follow the commands below to prepare the SBU-Shadow dataset:
```
$: pwd
/your_dir/cat-sam
$: cd ./data/sbu
$: wget https://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip
$: unzip SBU-shadow.zip
$: rm SBU-shadow.zip
$: conda activate cat-sam
(cat-sam) $: ls
preprocess.py  README.md  SBU-shadow
(cat-sam) $: python preprocess.py
(cat-sam) $: ls
preprocess.py  README.md  SBU-shadow  test.json  train.json
```

If you cannot download the dataset by `wget`, 
please press *'Download the SBU shadow dataset'* in the [official page](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html).

