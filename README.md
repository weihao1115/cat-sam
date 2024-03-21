# CAT-SAM: Conditional Tuning Network for Few-Shot Adaptation of Segmentation Anything Model


The official implementation of "CAT-SAM: Conditional Tuning Network for Few-Shot Adaptation of Segmentation Anything Model". 

[[arXiv](http://arxiv.org/abs/2402.03631)] [[Project Page](https://xiaoaoran.github.io/projects/CAT-SAM)]


Authors: [Aoran Xiao*](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=zh-EN), [Weihao Xuan*](https://scholar.google.com/citations?user=7e0W-2AAAAAJ&hl=en&authuser=1&oi=ao), [Heli Qi](https://scholar.google.co.jp/citations?user=CH-rTXsAAAAJ&hl=en), [Yun Xing](https://scholar.google.co.jp/citations?user=uOAYTXoAAAAJ&hl=en), [Ruijie Ren](https://scholar.google.com/citations?user=ce-2e8EAAAAJ&hl), [Xiaoqin Zhang](https://ieeexplore.ieee.org/author/37405025600), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en), [Shijian Lu](https://personal.ntu.edu.sg/shijian.lu/)

![image](https://github.com/weihao1115/CAT-SAM/blob/main/imgs/CAT-SAM.png)

## Prepare the virtual environment
Please git our project to your local machine and prepare our environment by the following commands:
```
$: cd cat-sam
$: conda env create -f environment.yaml
$: conda activate cat-sam
(cat-sam) $: python -m pip install -e .
```

## Prepare the datasets
Please refer to the README.md in the dataset-specific folders under `./data` to prepare each of them.


## Testing
For testing, please run:
```
$: cd cat-sam
$: pwd
/your_dir/cat-sam
$: conda activate cat-sam
(cat-sam) $: python test.py --dataset <your-target-dataset> --cat_type <your-target-type> --ckpt_path <your-target-ckpt>
```
For reproducing the results of CAT-SAM models in our paper, please download our checkpoints below to any place in your machine. 
You can refer to the one you are interested in by `--ckpt_path`.

**Note:** if you set `--dataset whu`, please prepare 1 x NVIDIA RTX A5000 (24GB) or the device with more or similar memory.

## Download Checkpoints:

To download the checkpoints, please visit the following Google Drive link:

[Google Drive](https://drive.google.com/drive/folders/1oik813aRkFvZh000GI_58TUsu1uW9LSF?usp=sharing)

## TODO List

- [x] Release of test code
- [ ] Release of training code