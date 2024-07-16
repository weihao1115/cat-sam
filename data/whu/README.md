# WHU Dataset

Please follow the commands below to prepare the WHU dataset:
```
$: pwd
/your_dir/cat-sam
$: cd ./data/whu
$: wget http://gpcv.whu.edu.cn/data/3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip
$: unzip '3. The cropped aerial image tiles and raster labels.zip'
$: rm '3. The cropped aerial image tiles and raster labels.zip'
$: conda activate cat-sam
(cat-sam) $: ls
'3. The cropped image tiles and raster labels'   preprocess.py   README.md
(cat-sam) $: python preprocess.py
(cat-sam) $: ls
'3. The cropped image tiles and raster labels'   preprocess.py   README.md   test.json   train.json
```

If you cannot download the dataset by `wget`, 
please download *'3.1 The cropped aerial image tiles and raster labels.zip (0.3 meter)'* in the [official page](http://gpcv.whu.edu.cn/data/building_dataset.html).
