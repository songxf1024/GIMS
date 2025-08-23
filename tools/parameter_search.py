"""
gims_search.py

GIMS参数搜索脚本

本脚本用于在指定图像对上进行大规模参数组合搜索，评估每组参数下图匹配的性能（正确匹配数 / 总匹配数 / 耗时）。
支持自动仿射变换图像生成、Homography 验证、匹配性能统计、批量保存结果等。

主要功能：
-----------
- 自动对 image0 执行仿射变换生成 image1（可选）
- 使用 CarHyNet 提取局部描述子
- 使用 Sinkhorn 进行图匹配
- 遍历 (r, t, m) 参数组合，评估匹配性能
- 保存 Excel 和 TXT 格式结果文件

参数说明：
-----------
- image0 / image1   ：两张输入图像路径，image1 留空可自动生成
- r/p/m             ：图构建参数（三元组）
  - r: 相邻点搜索半径
  - t: 相似度阈值百分位数
  - m: 子图最小节点数
- weights           ：CarHyNet 预训练权重路径
- output            ：结果保存目录
- max_keypoints     :保留前多少个点

依赖库：
-----------
- PyTorch
- OpenCV
- PIL
- torchvision
- pandas
- CarHyNet（自定义模块）
- Matching（自定义模块）

示例命令：
-----------
python gims_search.py \
  --image0 ./gims_search/image0.png \
  --image1 ./gims_search/image1.png \
  --cuda cuda:0 \
  --r-range 10,31 \
  --t-range 1,11 \
  --m-range 1,11 \
  --weights output/train/hynet/weights/minloss_our.pt

python gims_search.py \
  -i0 ./gims_search/image0.png \
  -c cuda:0 \
  -r 10,31 \
  -t 1,11 \
  -m 1,11

python gims_search.py \
  -i0 ./exp_gims_search/indoor0_0.png \
  -i1 ./exp_gims_search/indoor0_1.png


作者：
-----------
小锋学长生活大爆炸 | 2025
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil
import random
import argparse
import pandas as pd
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2
from .carhynet.models import HyNetnetFeature2D
from models.matching import Matching
import torchvision
import torch
import time


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def send_notify(msg):
    try:
        requests.get('http://wxbot.xfxuezhang.cn/send/friend?target=wxid_0438974384722&key=lihua&msg='+msg, timeout=10)
    except:
        pass

def gpu_warmup(cuda, num_warmup_steps=50):
    """对 GPU 进行预热"""
    model = torchvision.models.resnet18()
    input_tensor = torch.randn(1, 3, 224, 224)
    model = model.to(cuda)
    input_tensor = input_tensor.to(cuda)
    model.train()
    for _ in tqdm(range(num_warmup_steps), desc='GPU Warmup...', ncols=80):
        output = model(input_tensor)
        output.backward(torch.ones_like(output))
        torch.cuda.synchronize()
    del model

def heavy_affine_transform(image):
    transform = transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomAffine(
            degrees=150,
            translate=(0.3, 0.3),
            scale=(0.5, 1.5),
            shear=30
        )
    ])
    return transform(image)

def simple_affine_transform(image, heavy=False):
    if heavy: return heavy_affine_transform(image)
    angle = random.uniform(-90, 90)              # 更大角度旋转
    translate = (random.uniform(-0.3, 0.3) * image.width, random.uniform(-0.3, 0.3) * image.height)  # 更大范围的相对平移
    scale = random.uniform(0.5, 1.5)            # 缩放范围更极端
    shear = random.uniform(-30, 30)                # 强剪切
    return TF.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)

def safe_copy(src, dst):
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)

def search_process(param, image0_path, image1_path, device, hynet, matching, result_path):
    radius, percentile, min_size = param
    image0_name = image0_path.split("/")[-1].split(".")[0]
    image1_name = image1_path.split("/")[-1].split(".")[0]
    image0 = cv2.imread(image0_path, cv2.IMREAD_COLOR)
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    with torch.no_grad():
        pred = matching({'image0': np.expand_dims(image0, axis=0), 'image1': np.expand_dims(image1, axis=0), 'carhynet': hynet, 'device': device, 'radius': radius, 'percentile': percentile, 'min_size': min_size})
    t2 = time.perf_counter()
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    try:
        points1 = np.float32(mkpts0)
        points2 = np.float32(mkpts1)
        H, mask = cv2.findHomography(points1, points2, cv2.USAC_DEFAULT)
        result_count = len(points1[mask.ravel() == 1])
        total_count = len(matches)
        print(f"{param}, {image0_name}/{image1_name} => {result_count}/{total_count}")
        match_result = [param + [result_count, total_count, t2-t1]]
    except Exception as e:
        print("匹配的点数太少")
        match_result = [param + [0, 0, t2-t1]]
    finally:
        del image1
    save_file = os.path.join(result_path, 'record')
    columns = ['r', 't', 'm', 'correct_matches', 'total_matches', 'time']
    df_existing = pd.read_excel(save_file+'.xlsx') if os.path.exists(save_file+'.xlsx') else pd.DataFrame(columns=columns)
    df_new = pd.DataFrame(match_result, columns=columns)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_excel(save_file+'.xlsx', index=False)
    with open(save_file + '.txt', 'a+') as f: f.write(f'{str(match_result[0])}\n')

def search(image0_path, image1_path, parameters, weights_path, output_dir, max_keypoints=-1, cuda='cuda:0'):
    device = torch.device(cuda)
    config = {
        'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': -1},
        'superglue': {'weights_path': weights_path, 'sinkhorn_iterations': 20, 'match_threshold': 0.02},
        'max_keypoints': max_keypoints
    }
    hynet = HyNetnetFeature2D(do_cuda=(cuda!='cpu'), cuda=cuda)
    matching = Matching(config).eval().to(device)
    image0_name, image0_suffix = image0_path.split("/")[-1].split(".")
    image1_name, image1_suffix = image1_path.split("/")[-1].split(".") if image1_path else ('None', 'jpg')
    result_path = f'{output_dir}/{image0_name[:20]}_{image1_name[:20]}/'
    os.makedirs(result_path, exist_ok=True)
    # === 自动生成 image1 ===
    if image1_path is None:
        img0 = Image.open(image0_path).convert('RGB')
        img1 = simple_affine_transform(img0, heavy=False)
        image1_path = os.path.join(result_path, 'image1.jpg')
        img1.save(image1_path)
    # === 拷贝两张图到 result_path ===
    safe_copy(image0_path, os.path.join(result_path, f'{image0_name}.{image0_suffix}'))
    safe_copy(image1_path, os.path.join(result_path, image1_path.split("/")[-1]))
    try:
        for param in parameters:
            search_process(param, image0_path, image1_path, device, hynet, matching, result_path)
            torch.cuda.empty_cache()
    except Exception as e:
        send_notify(f"error:{image0_name}_{e}")
        raise e
    else:
        send_notify(f"success:{image0_name}_complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="GIMS")
    parser.add_argument("-i0", "--image0", type=str, default=None, help="Path to the first image (image0)")
    parser.add_argument("-i1", "--image1", type=str, default=None, help="Path to the second image (image1)")
    parser.add_argument("-c", "--cuda", type=str, default="cuda:0", help="CUDA device to use (e.g. cpu, cuda:0, cuda:2, cuda:4, cuda:6)")
    parser.add_argument("-r", "--r-range", type=str, default="10,30", help="Range for r (format: start,end)")
    parser.add_argument("-t", "--t-range", type=str, default="0,10", help="Range for t (format: start,end)")
    parser.add_argument("-m", "--m-range", type=str, default="0,10", help="Range for m (format: start,end)")
    parser.add_argument("-k", "--max-keypoints", type=int, default=-1, help="Max keypoints")
    parser.add_argument("-w", "--weights", type=str, default="./output/train/hynet/weights/minloss_our.pt", help="Path to pretrained weights")
    parser.add_argument("-o", "--output", type=str, default="./exp_gims_search", help="Dir to save results")
    return parser.parse_args()

def str_to_range(range_str):
    start, end = map(int, range_str.split(","))
    return list(range(start, end + 1))



if __name__ == "__main__":
    set_seed()
    # torch.set_grad_enabled(False)
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_args()
    r_range = str_to_range(args.r_range)
    t_range = str_to_range(args.t_range)
    m_range = str_to_range(args.m_range)
    parameters = [[r, t, m] for r in r_range for t in t_range for m in m_range]
    args.image0 = args.image0 or f'{args.output}/image0/image0.jpg'
    os.makedirs(args.output, exist_ok=True)
    print("\n===== GIMS Search Configuration =====")
    print(f"Image 0       : {args.image0}")
    print(f"Image 1       : {args.image1}")
    print(f"CUDA Device   : {args.cuda}")
    print(f"Weight Path   : {args.weights}")
    print(f"Save Dir      : {args.output}")
    print(f"r range       : {r_range}")
    print(f"t range       : {t_range}")
    print(f"m range       : {m_range}")
    print(f"Max Keypoints : {args.max_keypoints}")
    print(f"Total Params  : {len(parameters)} combinations")
    print("=====================================\n")

    gpu_warmup(args.cuda)
    search(
        image0_path=args.image0,
        image1_path=args.image1,
        parameters=parameters,
        weights_path=args.weights,
        output_dir=args.output,
        max_keypoints=args.max_keypoints,
        cuda=args.cuda
    )

'''
python parameter_search.py -c cuda:0 -i0 ../datasets/my/house/query.JPG -i1 ../datasets/my/house/h3.JPG
python parameter_search.py -c cuda:0 -i0 ./exp_gims_search/indoor0_0/indoor0_0.jpg -i1 ./exp_gims_search/indoor0_0/image1.jpg
python parameter_search.py -c cuda:6 -i0 ../datasets/my/public/boat/i1.png -i1 ../datasets/my/public/boat/i5.png
python parameter_search.py -c cuda:6 -i0 ../datasets/my/public/boat/i1.png -i1 ../datasets/my/public/boat/i6.png
python parameter_search.py -c cuda:2 -i0 ./exp_gims_search/outdoor0_0/outdoor0_0.jpg -i1 ./exp_gims_search/outdoor0_0/image1.jpg
python parameter_search.py -c cuda:2 -i0 ./exp_gims_search/indoor1_0/indoor1_0.jpg -i1 ./exp_gims_search/indoor1_0/image1.jpg
python parameter_search.py -c cuda:4 -i0 ./assets/coco_test_images/000000156570.jpg
python parameter_search.py -c cuda:4 -i0 ../datasets/coco/test2017/000000002271.jpg
python parameter_search.py -c cuda:2 -i0 ./exp_gims_search/000000123527/000000123527.jpg -i1 ./exp_gims_search/000000123527/image1.jpg
python parameter_search.py -c cuda:6 -i0 ./exp_gims_search/graf1.png ./exp_gims_search/graf3.png

python parameter_search.py -c cuda:0 -i0 ../datasets/my/house/query.JPG -i1 ../datasets/my/house/h2.JPG -k 10000 -r 0,50 -t 0,0 -m 0,0
'''

