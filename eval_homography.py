import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from pathlib import Path
import argparse
import random
import numpy as np
import torch
import cv2
from carhynet.models import HyNetnetFeature2D
from models.matching import Matching
from utils.common import AverageTimer, pose_auc, read_image_with_homography, \
                         compute_pixel_error, download_test_images, \
                         set_seed, send_notify
from utils.preprocess_utils import torch_find_matches


def draw_homography_boxes_ext(
        result_image,
        w0,
        homo_gt,
        homo_est,
        error_text=None,
        box_color_gt=(255, 0, 0),
        box_color_est=(0, 0, 255),
        thickness=3,
        font_scale=1.0,
        font_color=(0, 255, 0)
):
    """
    Draw on the right half of result_image (image1):
    - Blue border (GT homoography)
    - Red border (RANSAC estimate)
    - Optional error comment text (such as error_ransac)

    parameter:
        result_image: The output image
        w0: The width of image0 is used to locate the starting point of the right image
        homo_gt: Ground Truth Homography (3x3)
        homo_est: Estimated Homography (3x3)
        error_text: Optional text, such as "error: 2.34"
    return:
        Images with comments and borders
    """
    try:
        h, w = result_image.shape[:2]
        image1_w = w - w0
        image1_h = h
        corners = np.array([
            [0, 0],
            [0, image1_h],
            [image1_w, image1_h],
            [image1_w, 0]
        ], dtype=np.float32).reshape(-1, 1, 2)
        if homo_gt is not None:
            gt_box = cv2.perspectiveTransform(corners, homo_gt) + np.array([[[w0, 0]]], dtype=np.float32)
            gt_box = gt_box.astype(np.int32)
            cv2.polylines(result_image, [gt_box], isClosed=True, color=box_color_gt, thickness=thickness)
        if homo_est is not None:
            est_box = cv2.perspectiveTransform(corners, homo_est) + np.array([[[w0, 0]]], dtype=np.float32)
            est_box = est_box.astype(np.int32)
            cv2.polylines(result_image, [est_box], isClosed=True, color=box_color_est, thickness=thickness)
        # Add error text
        if error_text is not None:
            position = (w0 + 10, 30)  # On the upper left corner of the picture right
            cv2.putText(result_image, error_text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, font_color, thickness=2, lineType=cv2.LINE_AA)
    except Exception as e:
        print(f"[Error] Drawing borders or comments on result_image failed: {e}")
    return result_image

def draw_matches(img1, img2, matched_points1, matched_points2):
    def ensure_color(img):
        """Make sure the image is a three-channel color image. If it is a grayscale image, convert it to a color image."""
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR color image
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

def GIMS(dgims=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_homography', type=str, default='./assets/coco_test_images_homo.txt')
    parser.add_argument('--input_dir', type=str, default='./assets/coco_test_images/')
    parser.add_argument('--output_dir', type=str, default='./output/dump/dump_homo_pairs')
    parser.add_argument('--max_length', type=int, default=-1)
    parser.add_argument('--resize', type=int, nargs='+', default=[800, 600])
    parser.add_argument('--resize_float', action='store_true')
    parser.add_argument('--weights_path', default='./weights/gims_minloss_L.pt')
    parser.add_argument('--max_keypoints', type=int, default=-1)
    parser.add_argument('--sinkhorn_iterations', type=int, default=20)
    parser.add_argument('--min_matches', type=int, default=12)
    parser.add_argument('--match_threshold', type=float, default=0.02)
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument("--name", type=str, default="gims")
    parser.add_argument("--agc_r", type=int, default=15, help="radius")
    parser.add_argument("--agc_p", type=int, default=2, help="percentile")
    parser.add_argument("--agc_m", type=str, default=7, help="min_size")
    args = parser.parse_args()
    os.makedirs('./output/dump/', exist_ok=True)
    args.output_dir = args.output_dir + '_' + args.name
    print(args)

    if len(args.resize) == 2 and args.resize[1] == -1: args.resize = args.resize[0:1]
    if len(args.resize) == 2: print('resize to {}x{} (WxH)'.format(args.resize[0], args.resize[1]))
    elif len(args.resize) == 1 and args.resize[0] > 0: print('resize max dimension to {}'.format(args.resize[0]))
    elif len(args.resize) == 1: print('Will not resize images')
    else: raise ValueError('Cannot specify more than two integers for --resize')
    with open(args.input_homography, 'r') as f: homo_pairs = f.readlines()
    if args.max_length > -1: homo_pairs = homo_pairs[0:np.min([len(homo_pairs), args.max_length])]
    if args.shuffle: random.Random(0).shuffle(homo_pairs)
    download_test_images()
    cuda = 'cuda:0'
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    carhynet = HyNetnetFeature2D(cuda=cuda)
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'weights_path': args.weights_path,
        'sinkhorn_iterations': args.sinkhorn_iterations,
        'match_threshold': args.match_threshold,
        'max_keypoints': args.max_keypoints
    }
    matching = Matching(config).eval().to(device)
    input_dir = Path(args.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('write matches to directory \"{}\"'.format(output_dir))
    timer = AverageTimer(newline=True)
    results_file = []
    result_path = output_dir / 'result'
    result_name = result_path / 'results.txt'
    result_match_path = result_path / 'matches'
    os.makedirs(str(result_match_path), exist_ok=True)
    for i, info in enumerate(homo_pairs):
        split_info = info.strip().split(' ')
        image_name = split_info[0]
        homo_info = list(map(lambda x: float(x), split_info[1:]))
        homo_matrix = np.array(homo_info).reshape((3,3)).astype(np.float32)
        stem0 = Path(image_name).stem
        matches_path = output_dir / '{}_matches.npz'.format(stem0)
        eval_path = output_dir / '{}_evaluation.npz'.format(stem0)
        image0, image1, inp0, inp1, scales0, homo_matrix = read_image_with_homography(input_dir / image_name,
                                                                                      homo_matrix, device,
                                                                                      args.resize, 0,
                                                                                      args.resize_float,
                                                                                      color=True)
        if image0 is None or image1 is None: continue
        timer.update('load_image')
        with torch.no_grad():
            pred = matching({'delaunay': dgims, 'image0': inp0, 'image1': inp1, 'carhynet': carhynet, 'device': device,
                             'radius': args.agc_r, 'percentile': args.agc_p, 'min_size': args.agc_m})
        kp0_torch, kp1_torch = pred['keypoints0'][0], pred['keypoints1'][0]
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1, 'matches': matches, 'match_confidence': conf}
        np.savez(str(matches_path), **out_matches)
        valid = matches > -1
        mkpts0, mkpts1 = kpts0[valid], kpts1[matches[valid]]
        mconf = conf[valid]
        try:
            points1, points2 = np.float32(mkpts0), np.float32(mkpts1)
            H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)  # or cv2.USAC_DEFAULT
            print(f">> {len(points1[mask.ravel() == 1])}/{len(matches)}")
            results_file.append(f"{image_name} => {len(points1[mask.ravel()==1])}")
            matched_points1 = points1[mask.ravel() == 1]
            matched_points2 = points2[mask.ravel() == 1]
            result_image = draw_matches(image0, image1, matched_points1, matched_points2)
            # cv2.imshow('Matched Points', result_image)
            # cv2.waitKey(1)
            cv2.imwrite(str(result_match_path / image_name), result_image)
        except Exception:
            print(">> Too few points matched, skip")
            results_file.append(f"{image_name} => 0")
            cv2.imwrite(str(result_match_path / image_name), draw_matches(image0, image1, [], []))
            continue
        try:
            ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, torch.from_numpy(homo_matrix).to(kp0_torch.device), dist_thresh=3, n_iters=3)
        except Exception as e:
            print(">> skip", e)
            continue
        ma_0, ma_1 = ma_0.cpu().numpy(), ma_1.cpu().numpy()
        gt_match_vec = np.ones((len(matches), ), dtype=np.int32) * -1
        gt_match_vec[ma_0] = ma_1
        corner_points = np.array([[0,0], [0, image0.shape[0]], [image0.shape[1], image0.shape[0]], [image0.shape[1], 0]]).astype(np.float32)
        if len(mconf) < args.min_matches:
            out_eval = {'error_dlt': -1, 'error_ransac': -1, 'precision': -1, 'recall': -1}
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')
            print('Skipping {} due to inefficient matches'.format(i))
            continue
        sort_index = np.argsort(mconf)[::-1][0:4]
        est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
        est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)
        corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
        corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(1)
        corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), homo_matrix).squeeze(1)
        error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
        error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)
        match_flag = (matches[ma_0] == ma_1)
        precision = match_flag.sum() / valid.sum()
        fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
        recall = match_flag.sum() / (match_flag.sum() + fn_flag.sum())
        out_eval = {'error_dlt': error_dlt, 'error_ransac': error_ransac, 'precision': precision, 'recall': recall}
        np.savez(str(eval_path), **out_eval)
        cv2.imwrite(str(result_match_path / f"{stem0}_bordered.jpg"), draw_homography_boxes_ext(result_image, image0.shape[1], homo_matrix, est_homo_ransac, f"error_ransac: {error_ransac:.2f}"))
        timer.update('eval')
        timer.print('Finished pair {:5} of {:5}'.format(i, len(homo_pairs)))
    errors_dlt = []
    errors_ransac = []
    precisions = []
    recall = []
    for info in homo_pairs:
        split_info = info.strip().split(' ')
        image_name = split_info[0]
        stem0 = Path(image_name).stem
        eval_path = output_dir / '{}_evaluation.npz'.format(stem0)
        if os.path.exists(eval_path):
            results = np.load(eval_path)
            if results['precision'] == -1: continue
            errors_dlt.append(results['error_dlt'])
            errors_ransac.append(results['error_ransac'])
            precisions.append(results['precision'])
            recall.append(results['recall'])
    thresholds = [5, 10, 25]
    aucs_dlt = pose_auc(errors_dlt, thresholds)
    aucs_ransac = pose_auc(errors_ransac, thresholds)
    aucs_dlt = [100.*yy for yy in aucs_dlt]
    aucs_ransac = [100.*yy for yy in aucs_ransac]
    prec = 100.*np.mean(precisions)
    rec = 100.*np.mean(recall)
    print('Evaluation Results (mean over {} pairs):'.format(len(homo_pairs)))
    print("For DLT results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs_dlt[0], aucs_dlt[1], aucs_dlt[2], prec, rec))
    print("For homography results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs_ransac[0], aucs_ransac[1], aucs_ransac[2], prec, rec))
    with open(result_name, 'w+') as f: f.write('\n'.join(results_file))
    send_notify('Done：'+args.output_dir)

if __name__ == '__main__':
    # python eval_homography.py --input_homography assets/coco_test_images_homo.txt --input_dir assets/coco_test_images --weights_path ./weights/minloss.pt --name=gims
    set_seed(42)
    GIMS(dgims=False)

