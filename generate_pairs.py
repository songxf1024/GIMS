import os
import numpy as np
import cv2
import time
from utils.preprocess_utils import get_perspective_mat
import os
import shutil
import random
from tqdm import tqdm

random.seed(3)
np.random.seed(3)


def move(total_num=500):
    source_folder = '../datasets/rgbd/outdoor_test'
    target_folder = f"mydatasets/outdoor_{total_num}_images/input"

    if not os.path.exists(target_folder): os.makedirs(target_folder)
    files = [f for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]
    selected_files = files if len(files) < total_num else random.sample(files, total_num)
    for file in tqdm(selected_files):
        shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))
    print("file copy complete.")


def process():
    image_dir = "mydatasets/outdoor_500_images/input"
    txt_file = open("mydatasets/outdoor_500_images.txt", 'w') #path where the generated homographies should be stored
    image_save_path = "mydatasets/outdoor_500_images/output" #path where the original and warped image will be stored for visualization
    if not os.path.isdir(image_save_path): os.makedirs(image_save_path)
    content = os.listdir(image_dir)
    ma_fn = lambda x: float(x)
    for kk, i in enumerate(tqdm(content)):
        if os.path.splitext(i)[-1] not in [".jpg", ".png"]: continue
        image = cv2.imread(os.path.join(image_dir, i))
        height, width = image.shape[:2]
        # all the individual perspective component range should be adjusted below
        homo_matrix = get_perspective_mat(0.85,center_x=width//2, center_y=height//2, pers_x=0.0008, pers_y=0.0008, shear_ratio=0.04, shear_angle=10, rotation_angle=25, scale=0.6, trans=0.6)
        res_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
        txt_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(i, *list(map(ma_fn, list(homo_matrix.reshape(-1))))))
        write_img = np.concatenate([image, res_image], axis=1)
        cv2.imwrite(os.path.join(image_save_path, "{}.png".format(kk+1)), write_img)


def random_generate(cnt=1):
    image_path = "mydatasets/random/0.jpg"
    image_save_path = "mydatasets/random/"
    image = cv2.imread(image_path)
    for i in tqdm(range(cnt)):
        height, width = image.shape[0:2]
        homo_matrix = get_perspective_mat(0.85,
                                          center_x=width // 2, center_y=height // 2,
                                          pers_x=0.0008, pers_y=0.0008,
                                          shear_ratio=0.04, shear_angle=10,
                                          rotation_angle=25, scale=0.6, trans=0.6)
        res_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
        cv2.imwrite(os.path.join(image_save_path, "{}.png".format(i+1)), res_image)


if __name__ == '__main__':
    move()
    process()
    # random_generate(cnt=100)


