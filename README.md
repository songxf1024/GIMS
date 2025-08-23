# Image Matching System Based on Adaptive Graph Construction and Graph Neural Network

---
    

## Directory Structure
Below is a brief overview of the directory structure and the role of each component:
```bash
gims/                     # Source code for GIMS framework
├── assets/               # Example images for demo and visualization
├── carhynet/             # CAR-HyNet descriptor implementation
├── configs/              # Training configuration files
├── models/               # AGC and GNN matcher
├── tools/                # Some auxiliary tools
├── utils/                # Common utility functions and helpers
├── weights/              # Pretrained weights for GIMS and CAR-HyNet
├── generate_pairs.py     # Script to generate image pairs for training/evaluation
├── train.py              # Training entry point
├── eval_homography.py    # Homography estimation and AUC evaluation
├── eval_match.py         # Matching evaluation and statistics
└── requirements.txt      # Python dependencies
```

> - The weights of GIMS and CAR-HyNet can be downloaded at [Google Drive](https://kutt.it/gims)  
> - Online GIMS Parameter Analysis Dashboard: [https://gims.xfxuezhang.cn/](https://gims.xfxuezhang.cn/) or [http://14.103.144.178/](http://14.103.144.178/)

## Setup
### Environment
#### Hardware Dependencies
> The following configurations are recommended and not mandatory.
- X86-CPU machine (>= 16 GB RAM) 
- Nvidia GPUs (> 10 GB each)

#### Software Dependencies
- Ubuntu 20.04 LTS
- Python 3.9.21
- CUDA 11.7
- PyTorch 2.0.1
- DGL 1.1.2

### Installation
Running the following commands will create the virtual environment and install the dependencies. `conda` can be downloaded from [anaconda](https://www.anaconda.com/download/success). 
```bash
conda create -n gims python=3.9.12
conda activate gims

pip install -r requirements
```

### Datasets
- We use [COCO2017](https://cocodataset.org/#download) for training. Download the 'train2017', 'val2017', and 'annotations' folder and put the folder path in the config file.
- We use [COCO2017 (Test images)](http://images.cocodataset.org/zips/test2017.zip), [DIML RGB-D](https://dimlrgbd.github.io/), and [Oxford-Affine](https://www.robots.ox.ac.uk/~vgg/research/affine/) for evaluation. 

## Usage
### Experiment Customization
Adjust configurations in `configs/coco_config.yaml` to customize  train params, optimizer params, and dataset params. In general, you only need to modify `dataset_params.dataset_path`.

### Train the Model
Running the following command to start training the model.
```bash
python train.py --gpus="0" --limit=-1 --name=gims
```
The output in the console will be like:
```bash
GPU 0: NVIDIA GeForce RTX 3090
==> CAR-HyNet successfully loaded pre-trained network.
Optimizer groups: 139 .bias, 120 conv.weight, 23 other
loading annotations into memory...
Done (t=18.04s)
creating index...
index created!
Started training for 2 epochs
Number of batches: 118286
Chang learning rate to 0.0001
Started epoch: 1 in rank -1
Epoch   gpu_mem   Iteration   PosLoss   NegLoss   TotLoss     Dtime     Ptime     Mtime
  0      0.463G       0        3.28        0       3.28      0.06663    11.43     3.069:   0%|    | 1/118286 [00:14<479:21:19, 14.59s/it]
  0      0.445G       1        3.491       0       3.49      0.03858    6.245     2.511:   0%|    | 2/118286 [00:17<255:55:56,  7.79s/it]
```

### Training Arguments
Core training arguments are listed below:
```bash
--config_path: the path of config.yaml
--backend: the backend of communication, use NCCL
--gpus: the ids of GPUs, e.g., '0' means one GPU, '0,2' means two GPUs
--name: the folder name for output data, e.g., output/train/<name>
--limit: number of training images. -1 means all
```
Please refer to `train.py` for more details. 

### Generate Image Pairs
Running the following command to generate image pairs with random homographies.
```bash
python generate_pairs.py
```
This script first moves a certain number of images from the source folder to the target folder, and then generates images pairs using random homographies. Finally, the **image pairs** and the **homography matrix** will be saved. Please refer to `generate_pairs.py` for more details.

### Evaluation of AUC
Running the following command to evaluate the number of correct matches.
```bash
python eval_homography.py
```
The results will be saved in `output/dump/`. Please refer to `eval_homography.py` for more details.

The output in the console will be like:
```bash
resize to 800x600 (WxH)
==> CAR-HyNet successfully loaded pre-trained network.
Running inference on device "cuda:0"
Loaded GMatcher model ("./weights/gims_minloss_L.pt" weights)
Looking for data in directory "./mydatasets/test_images"
write matches to directory "output/dump/dump_homo_pairs_gims"
[Finished pair     0 of   199] load_image=0.009 matcher=17.216 eval=0.088 total=17.313 sec {0.1 FPS} 
[Finished pair     1 of   199] load_image=0.009 matcher=15.254 eval=0.076 total=15.339 sec {0.1 FPS} 
[Finished pair     2 of   199] load_image=0.008 matcher=15.157 eval=0.103 total=15.268 sec {0.1 FPS} 
...
[Finished pair   198 of   199] load_image=0.008 matcher=14.195 eval=0.089 total=14.293 sec {0.1 FPS} 
Evaluation Results (mean over 199 pairs):
For DLT results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
1.54     3.48    7.04    35.42   76.76  
For homography results...
AUC@5    AUC@10  AUC@25  Prec    Recall 
76.44    85.52   91.47   35.42   76.76 
```

### Evaluation of MN
Running the following command to evaluate the number of correct matches.
```bash
python eval_matches.py
```
- You can change `image0_path` and `image1s_path` in `eval_matches.py` to evaluate different image pairs.
- Note that `image1s_path` can be a single image path or a list of multiple image paths.
- You can set `dgims=True` to evaluate `D-GIMS`, which uses `Delaunay triangulation` to construct the graph. 

The results will be saved in `output/match/`. Please refer to `eval_matches.py` for more details.

The output in the console will be like:
```bash
GPU Warmup...: 100%|████████████████████████████| 50/50 [00:00<00:00, 50.26it/s]
NVIDIA GeForce RTX 3090
==> CAR-HyNet successfully loaded pre-trained network.
Loaded GMatcher model ("./weights/gims_minloss_L.pt" weights)
---------------------------
[i1/i2]
>> Keypoint Detection: 0.11635971069335938
>> Number of Keypoints: 15382
>> Patches Generation: 3.2201309204101562
>> Descriptor Generation: 0.6913251876831055
>> Keypoint Detection: 0.0947268009185791
>> Number of Keypoints: 14870
>> Patches Generation: 3.876391649246216
>> Descriptor Generation: 0.4590010643005371
>> Graph Construction 1: 5.720302104949951
>> Graph Construction 2: 6.192043304443359
>> Image Matching: 3.4824626445770264
!! Peak GPU memory: 7.38 GB
>> RANSAC: 0.07443380355834961
>> Total Time: 24.042573928833008
5392/15382
```


## License
Copyright (c) 2025 xxx. All rights reserved.  
Licensed under the MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=songxf1024/GIMS&type=Date)](https://www.star-history.com/#songxf1024/GIMS&Date)
