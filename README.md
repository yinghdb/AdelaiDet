# EmbedMask

    EmbedMask: Embedding Coupling for One-stage Instance Segmentation;
    Hui Ying, Zhaojin Huang, Shu Liu, Tianjia Shao and Kun Zhou;
    arXiv preprint arXiv:1912.01954

[[`Paper`](https://arxiv.org/abs/1912.01954)]


## Installation
This EmbedMask implementation is based on [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), which is also on top of [Detectron2](https://github.com/facebookresearch/detectron2). 

First, follow the [default instruction](../../README.md#Installation) to install the project and [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) 
set up the datasets (e.g., MS-COCO).

## Pretrained Models
The pretrained models can be downloaded from [here](https://1drv.ms/u/s!Al_gruIFwTUskjUekAWxezmkUnZf?e=c9Ymn0). And you should place them in the 'models' directory.

## Demo
For demo, run the following command lines:
```
python demo/demo.py \
    --config-file configs/EmbedMask/MS_R_101_3x.yaml \
    --input demo/images \
    --output demo/outputs \
    --opts MODEL.WEIGHTS models/EmbedMask_R_101_3x.pth
```


## Evaluation
For evaluation on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/EmbedMask/MS_R_50_1x.yaml \
    --eval-only \
    --num-gpus 4 \
    OUTPUT_DIR training_dir/EmbedMask_R_50_1x \
    MODEL.WEIGHTS models/EmbedMask_R_50_1x.pth
```

## Training
For training on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/EmbedMask/MS_R_50_1x.yaml \
    --num-gpus 4 \
    OUTPUT_DIR training_dir/EmbedMask_R_50_1x
```

## Results

Name | box AP (val) | mask AP (val) | box AP (test-dev) | mask AP (test-dev) 
--- |:---:|:---:|:---:|:---:
EmbedMask_MS_R_50_1x | 39.9 | 36.2 | 40.1 | 36.3
EmbedMask_MS_R_101_3x | 44.2 | 39.5 | 44.6 | 40.0 

## Notation

The main network architecture in this implementation is similar with that of CondInst and the auxiliary semantic segmentation task is used to help with the mask prediction.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@misc{ying2019embedmask,
    title={EmbedMask: Embedding Coupling for One-stage Instance Segmentation},
    author={Hui Ying and Zhaojin Huang and Shu Liu and Tianjia Shao and Kun Zhou},
    year={2019},
    eprint={1912.01954},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```