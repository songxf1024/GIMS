import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import time
from glob import glob
import cv2
import numpy as np
import torch
from carhynet.models import HyNetnetFeature2D
from models.matching import Matching
from utils.common import gpu_warmup, set_seed, send_notify


def calculate_nndr(descriptor_a, descriptor_b, threshold=0.8):
    # If the input is a numpy array, it is converted to a torch tensor
    if isinstance(descriptor_a, np.ndarray):
        descriptor_a = torch.tensor(descriptor_a)
    if isinstance(descriptor_b, np.ndarray):
        descriptor_b = torch.tensor(descriptor_b)
    # Transpose to ensure that the shape is (number, feature dimension)
    descriptor_a = descriptor_a.t()
    descriptor_b = descriptor_b.t()
    # Calculate the Euclidean distance between each descriptor in A and all descriptors in B
    distances = torch.cdist(descriptor_a, descriptor_b)
    #For each descriptor in A, find the descriptor in B, which is closest and second closest, B,
    distances_sorted, indices = torch.sort(distances, dim=1)
    nearest_distances = distances_sorted[:, 0]
    second_nearest_distances = distances_sorted[:, 1]
    # Calculate the distance ratio between the nearest neighbor and the second neighbor
    ratios = nearest_distances / second_nearest_distances
    # Find a good match based on the threshold
    matches = ratios < threshold
    # Returns the matching index and ratio
    match_indices = matches.nonzero().squeeze()
    good_matches = indices[match_indices, 0]  # Get the index of the nearest neighbor
    return match_indices, good_matches, ratios[matches]

def calculate_mnn(descriptor_a, descriptor_b, threshold=0.8):
    # If the input is a numpy array, it is converted to a torch tensor
    if isinstance(descriptor_a, np.ndarray):
        descriptor_a = torch.tensor(descriptor_a)
    if isinstance(descriptor_b, np.ndarray):
        descriptor_b = torch.tensor(descriptor_b)
    # Transpose to ensure that the shape is (number, feature dimension)
    descriptor_a = descriptor_a.t()
    descriptor_b = descriptor_b.t()
    # Calculate the Euclidean distance between each descriptor in A and all descriptors in B
    distances_ab = torch.cdist(descriptor_a, descriptor_b)
    distances_ba = torch.cdist(descriptor_b, descriptor_a)
    # For each descriptor in A, find the descriptor in B, which is closest and second closest, B,
    distances_sorted_ab, indices_ab = torch.sort(distances_ab, dim=1)
    nearest_ab = indices_ab[:, 0]
    second_nearest_ab = indices_ab[:, 1]
    # For each descriptor in B, find the descriptor in A which is closest and second closest to A
    distances_sorted_ba, indices_ba = torch.sort(distances_ba, dim=1)
    nearest_ba = indices_ba[:, 0]
    second_nearest_ba = indices_ba[:, 1]
    # Calculate mutual nearest neighbors
    mutual_nn_mask = torch.arange(descriptor_a.size(0)).to(descriptor_a.device) == nearest_ba[nearest_ab]
    # Find a good match based on the threshold
    nearest_distances = distances_sorted_ab[:, 0]
    second_nearest_distances = distances_sorted_ab[:, 1]
    ratios = nearest_distances / second_nearest_distances
    matches = (ratios < threshold) & mutual_nn_mask
    # Returns the matching index and ratio
    match_indices = matches.nonzero().squeeze()
    good_matches = nearest_ab[match_indices]  # Get the index of mutual nearest neighbors
    return match_indices, good_matches, ratios[matches]

def fine_match(points1, points2):
    try:
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        mkpts0 = points1[mask.ravel() == 1]
        mkpts1 = points2[mask.ravel() == 1]
        return mkpts0, mkpts1
    except:
        return [], []

def find_matching_keypoints(kpts0, kpts1, match_indices, good_matches):
    if(good_matches.numel() == 0):
        return [], []
    # Find the key points of matching from kpts0 using matching index
    matched_kpts0 = [kpts0[idx] for idx in match_indices.cpu().numpy()]
    # Use good_matches to find matching key points from kpts1
    matched_kpts1 = [kpts1[idx] for idx in good_matches.cpu().numpy()]
    return matched_kpts0, matched_kpts1

def draw_matches(img1, img2, matched_points1, matched_points2):
    def ensure_color(img):
        """Make sure the image is a three-channel color image. If it is a grayscale image, convert it to a color image."""
        if len(img.shape) == 2:  # Grayscale (height and width only)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR color image
        return img

    img1 = ensure_color(img1)
    img2 = ensure_color(img2)
    # Create a new image with a width of the sum of the widths of the two images and a height of the maximum between them
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    new_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')

    # Place two images on a new image
    new_image[:h1, :w1] = img1
    new_image[:h2, w1:w1 + w2] = img2

    # Draw lines on the new image for each pair of matching key points
    for p1, p2 in zip(matched_points1, matched_points2):
        start_point = (int(p1[0]), int(p1[1]))
        end_point = (int(p2[0] + w1), int(p2[1]))
        cv2.line(new_image, start_point, end_point, (0, 0, 255), 1)
        cv2.circle(new_image, start_point, 2, (0, 255, 0), -1)
        cv2.circle(new_image, end_point, 2, (255, 0, 0), -1)

    # Show the number of black text matches with white background in the upper left corner
    matches_count = len(matched_points1)
    text = f"Matches: {matches_count}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_offset_x, text_offset_y = 10, 30
    box_coords = (
    (text_offset_x, text_offset_y + 10), (text_offset_x + text_width, text_offset_y - text_height - 10))
    cv2.rectangle(new_image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(new_image, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return new_image

def GIMS(image0_path, image1s_path, root_path='output/match/', dgims=False, save_match=False):
    radius, percentile, min_size = 15, 2, 7
    cuda = 'cuda:0'
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device))
    result_match_path = root_path + ('dgims/' if dgims else 'gims/')
    result_name = result_match_path + 'result.txt'
    os.makedirs(result_match_path, exist_ok=True)
    weights_path = './weights/gims_minloss_L.pt'
    config = {
        'weights_path': weights_path,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.02,
        'max_keypoints': -1
    }
    carhynet = HyNetnetFeature2D(cuda=cuda)
    matching = Matching(config).eval().to(device)
    results_file = []
    for image1_path in glob(image1s_path):
        t_start = time.time()
        image0_name = image0_path.split("/")[-1].split(".")[0]
        image1_name = image1_path.split("/")[-1].split(".")[0]
        if image0_name == image1_name: continue
        print(f'---------------------------\n[{image0_name}/{image1_name}]')
        image0 = cv2.imread(image0_path, cv2.IMREAD_COLOR)
        image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            pred = matching({'delaunay': dgims, 'image0': np.expand_dims(image0, axis=0), 'image1': np.expand_dims(image1, axis=0), 'carhynet': carhynet, 'device': device, 'radius': radius, 'percentile': percentile, 'min_size': min_size})
        print(f"!! Peak GPU memory: {torch.cuda.max_memory_allocated()/(1024 ** 3):.2f} GB")
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        try:
            points1 = np.float32(mkpts0)
            points2 = np.float32(mkpts1)
            t1 = time.time()
            H, mask = cv2.findHomography(points1, points2, cv2.USAC_DEFAULT)
            print('>> RANSAC:', time.time() - t1)
            t_total = time.time() - t_start
            print('>> Total Time:', t_total)
            print(f"{len(points1[mask.ravel() == 1])}/{len(matches)}")
            result_count = len(points1[mask.ravel() == 1])
            results_file.append(f"{image1_name} => {result_count} [{t_total}]")
            if save_match:
                matched_points1 = points1[mask.ravel() == 1]
                matched_points2 = points2[mask.ravel() == 1]
                result_image = draw_matches(image0, image1, matched_points1, matched_points2)
                cv2.imwrite(result_match_path + image1_name + '.jpg', result_image)
        except Exception as e:
            print("Too few points matched:", e)
            results_file.append(f"{image1_name} => 0")
            continue
    with open(result_name, 'w+') as f: f.write('\n'.join(results_file))
    send_notify('eval_matches complete!')


if __name__ == '__main__':
    set_seed()
    image0_path = './datasets/my/public/boat/i1.png'
    image1s_path = './datasets/my/public/boat/i2.png'
    root_path = './output/match/boat/'
    gpu_warmup('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    GIMS(image0_path, image1s_path, root_path=root_path, dgims=False, save_match=True)
