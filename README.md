<h1 align="center">CAT-SAM: Conditional Tuning for Few-Shot Adaptation of Segment Anything Model
</h1>
<p align="center">
<a href="[https://arxiv.org/abs/2402.03631](https://arxiv.org/abs/2402.03631)"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/2402.03631">Conditional Tuning Network for Few-Shot Adaptation of Segmentation Anything Model</a>.</h4>
<h5 align="center">
    <em>
        <a href="https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en">Aoran Xiao*</a>,
        <a href="https://scholar.google.com/citations?user=7e0W-2AAAAAJ&hl=en">Weihao Xuan*</a>,
        <a href="https://scholar.google.co.jp/citations?user=CH-rTXsAAAAJ&hl=en">Heli Qi</a>,
        <a href="https://scholar.google.co.jp/citations?user=uOAYTXoAAAAJ&hl=en">Yun Xing</a>,
        <a href="https://scholar.google.com/citations?user=ce-2e8EAAAAJ&hl">Ruijie Ren</a>,
        <a href="https://ieeexplore.ieee.org/author/37405025600">Xiaoqin Zhang</a>,
        <a href="https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en">Ling Shao</a>,
        <a href="https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en">Shijian Lu</a>  (* indicates co-first authors with equal contributions.)
    </em>
</h5><p align="center">

## About
The Segment Anything Model (SAM) has demonstrated remarkable zero-shot capability and flexible geometric prompting in general image segmentation. However, it often struggles in domains that are either sparsely represented or lie outside its training distribution, such as aerial, medical, and non-RGB images. Recent efforts have predominantly focused on adapting SAM to these domains using fully supervised methods, which necessitate large amounts of annotated training data and pose practical challenges in data collection. This paper presents CAT-SAM, a ConditionAl Tuning network that explores _few-shot adaptation_ of SAM toward various challenging downstream domains in a data-efficient manner. The core design is a _prompt bridge_ structure that enables _decoder-conditioned joint tuning_ of the heavyweight image encoder and the lightweight mask decoder. The bridging maps the domain-specific features of the mask decoder to the image encoder, fostering synergic adaptation of both components with mutual benefits with few-shot target samples only, ultimately leading to superior segmentation in various downstream tasks. We develop two CAT-SAM variants that adopt two tuning strategies for the image encoder: one injecting learnable prompt tokens in the input space and the other inserting lightweight adapter networks. Extensive experiments over 11 downstream tasks show that CAT-SAM achieves superior segmentation consistently even under the very challenging one-shot adaptation setup.


## News
- **(2024/7)** We released the training code. Thank you for your waiting!
- **(2024/6)** CAT-SAM is accepted by ECCV 2024! See you in Milano!


## Method
![overall_pipeline](./figs/CAT-SAM.png "overall_pipeline")

## Results
1-Shot Adaptation:

| Methods        | WHU  | Kvasir | SBU-Shadow | Average |
|----------------|------|--------|------------|---------|
| **SAM (baseline)** | 43.5 | 79.0   | 62.4       | 61.6    |
| **VPT-shallow**   | 60.8 | 79.8   | 68.7       | 69.8    |
| **VPT-deep**      | 57.8 | 80.4   | 76.0       | 71.4    |
| **AdaptFormer**   | 83.2 | 76.8   | 77.2       | 79.1    |
| **LoRA**          | 86.1 | 77.5   | 74.4       | 79.3    |
| **CAT-SAM-T (Ours)**  | 86.8 | 83.4   | 78.0       | 82.7    |
| **CAT-SAM-A (Ours)**  | 88.2 | 85.4   | 81.9       | 85.2    |

![Visualizations](./figs/CAT-SAM.png "overall_pipeline")

## Installation
Please clone our project to your local machine and prepare our environment by the following commands:
```
$: cd cat-sam
$: conda env create -f environment.yaml
$: conda activate cat-sam
(cat-sam) $: python -m pip install -e .
```

The code has been tested on A100/A6000/V100 with Python 3.9, CUDA 11.7 and Pytorch 1.13.1. Any other devices may require to update the code for compatibility.



## Prepare the datasets
Please refer to the README.md in the dataset-specific folders under `./data` to prepare each of them.

## Train
Before training, please download the SAM checkpoints to `./pretrained` from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints).
After downloading, there should be:
```
cat-sam/
    pretrained/
        sam_vit_b_01ec64.pth
        sam_vit_h_4b8939.pth
        sam_vit_l_0b3195.pth
    ...
```

For one-shot training, please run:
```
$: cd cat-sam
$: pwd
/your_dir/cat-sam
$: conda activate cat-sam
(cat-sam) $: python train.py --dataset <your-target-dataset> --cat_type <your-target-type> --shot_num 1
```
Please prepare the following GPUs according to the following conditions:
1. `--dataset whu`: 1 x NVIDIA RTX A6000 (48GB) or ones with similar memories
2. `--dataset <other-than-whu>`: 1 x NVIDIA RTX A5000 (24GB) or ones with similar memories

After running, the checkpoint with the best performance will be automatically saved to `./exp/{dataset}_{sam_type}_{cat_type}_1shot`.
For the valid inputs for the arguments, please refer to the help message:
```
$: python train.py --help
```

For 16-shot and full-shot training, please run:
```
$: cd cat-sam
$: pwd
/your_dir/cat-sam
$: conda activate cat-sam
(cat-sam) $: CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset <your-target-dataset> --cat_type <your-target-type> (--shot_num 16)
```
If the argument `--shot_num` is not specified, training will proceed with the full-shot condition. 
Please prepare the following GPUs according to the following conditions:
1. `--dataset whu`: 4 x NVIDIA RTX A6000 (48GB) or ones with similar memories
2. `--dataset <other-than-whu>`: 4 x NVIDIA RTX A5000 (24GB) or ones with similar memories


## Test

We provide the checkpoints for the 1-shot experiments.
To download the checkpoints, please visit [here](https://drive.google.com/drive/folders/1oik813aRkFvZh000GI_58TUsu1uW9LSF?usp=sharing).

For testing, please run:
```
$: cd cat-sam
$: pwd
/your_dir/cat-sam
$: conda activate cat-sam
(cat-sam) $: python test.py --dataset <your-target-dataset> --cat_type <your-target-type> --ckpt_path <your-target-ckpt>
```



## Citation
If you find this work helpful, please kindly consider citing our paper:
```bibtex
@article{catsam,
  title={CAT-SAM: Conditional Tuning for Few-Shot Adaptation of Segment Anything Model},
  author={Xiao, Aoran and Xuan, Weihao and Qi, Heli and Xing, Yun and Ren, Ruijie and Zhang, Xiaoqin and Shao, Ling and Lu, Shijian},
  journal={arXiv preprint arXiv:2402.03631},
  year={2024}
}
```


## Acknowledgement
We acknowledge the use of the following public resources throughout this work: [Segment Anything Model](https://github.com/facebookresearch/segment-anything), [HQ-SAM](https://github.com/SysCV/sam-hq), [EVP](https://github.com/NiFangBaAGe/Explicit-Visual-Prompt) and [VPT](https://github.com/kmnp/vpt).


