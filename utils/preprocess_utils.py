import os
import numpy as np
import cv2
import torch

def get_rotmat(angle, as_3d=False, scale=1.0, center_x=0.0, center_y=0.0):
    cos_angle, sine_angle = np.cos(angle) * scale, np.sin(angle) * scale
    rotation = [cos_angle, -sine_angle, sine_angle, cos_angle]
    rotation = np.reshape(rotation, (2, 2)).T
    if as_3d:
        matrix_3d = np.eye(3)
        matrix_3d[:2, :2] = rotation
        matrix_3d[0, 2] = ((1 - cos_angle)*center_x) - (sine_angle*center_y)
        matrix_3d[1, 2] = (sine_angle*center_x) + ((1-cos_angle)*center_y)
        return matrix_3d
    return rotation

def get_translation_mat(image_height, image_width, trans, transformed_corners):
    left_top_min = np.min(transformed_corners, axis=0)
    right_bottom_min = np.min(np.array([image_width, image_height]) - transformed_corners, axis=0)
    trans_x_value = int(np.random.uniform(0, trans) * image_width)
    trans_y_value = int(np.random.uniform(0, trans) * image_height)
    if np.random.uniform() > 0.5: #translate x with respect to left axis
        trans_x = trans_x_value if left_top_min[0] < 0 else -trans_x_value
    else: #translate x with respect to right axis
        trans_x = trans_x_value if right_bottom_min[0] > 0 else -trans_x_value
    if np.random.uniform() > 0.5: #translate y with respect to top axis
        trans_y = trans_y_value if left_top_min[1] < 0 else -trans_y_value
    else: #translate y with respect to bottom axis
        trans_y = trans_y_value if right_bottom_min[1] > 0 else -trans_y_value
    translate_mat = np.eye(3)
    translate_mat[0, 2] = trans_x
    translate_mat[1, 2] = trans_y
    return translate_mat

def get_perspective_mat(patch_ratio, center_x, center_y, pers_x, pers_y, shear_ratio, shear_angle, rotation_angle, scale, trans):
    # Convert the input parameters to radians
    shear_angle, rotation_angle = np.deg2rad(shear_angle), np.deg2rad(rotation_angle)
    # Calculate the height and width of the image
    image_height, image_width = center_y * 2, center_x * 2
    # Calculate the boundary of the patch based on the given ratio
    patch_bound_w, patch_bound_h = int(patch_ratio * image_width), int(patch_ratio * image_height)
    # Create four corner coordinates of patch
    patch_corners = np.array([[0,0], [0, patch_bound_h], [patch_bound_w, patch_bound_h], [patch_bound_w, 0]]).astype(np.float32)
    # Randomly generate values ​​in the x and y directions of the perspective matrix
    pers_value_x = np.random.normal(0, pers_x/2)
    pers_value_y = np.random.normal(0, pers_y/2)
    pers_matrix = np.array([[1, 0, 0], [0, 1, 0], [pers_value_x, pers_value_y, 1]])
    # Generate shear matrix based on the given shear_ratio value and random number
    # shear_ratio is given by shear_x/shear_y
    if np.random.uniform() > 0.5:
        shear_ratio_value = np.random.uniform(1, 1+shear_ratio)
        shear_x, shear_y = 1, 1 / shear_ratio_value
    else:
        shear_ratio_value = np.random.uniform(1-shear_ratio, 1)
        shear_x, shear_y = shear_ratio_value, 1
    shear_angle_value = np.random.uniform(-shear_angle, shear_angle)
    shear_matrix = get_rotmat(-shear_angle_value, as_3d=True, center_x=center_x, center_y=center_y) @ np.diag([shear_x, shear_y, 1]) @ get_rotmat(shear_angle_value, as_3d=True, center_x=center_x, center_y=center_y)
    shear_perspective = shear_matrix @ pers_matrix
    # Randomly generate rotation angle and scaling
    rotation_angle_value = np.random.uniform(-rotation_angle, rotation_angle)
    scale_value = np.random.uniform(1, 1+(2*scale))
    # priotrising scaling up compared to scaling down
    scaled_rotation_matrix = get_rotmat(rotation_angle_value, as_3d=True, scale=scale_value, center_x=center_x, center_y=center_y)
    # Calculate the final homography matrix
    homography_matrix = scaled_rotation_matrix @ shear_perspective
    # Transform the corner coordinates of patch to a new position through perspective transformation and translation operations
    trans_patch_corners = cv2.perspectiveTransform(np.reshape(patch_corners, (-1, 1, 2)), homography_matrix).squeeze(1)
    translation_matrix = get_translation_mat(image_height, image_width, trans, trans_patch_corners)
    # Finally return to the homography matrix
    homography_matrix = translation_matrix @ homography_matrix
    return homography_matrix

def torch_cdist(keypoints1, keypoints2):
    diff = (keypoints1[:, None, :] - keypoints2[None, :, :]) ** 2
    summed = diff.sum(-1)
    distance = torch.sqrt(summed)
    return distance

def torch_setdiff1d(miss_index, match_index):
    combined = torch.cat((miss_index, match_index))
    unq, count = combined.unique(return_counts=True)
    diff = unq[count == 1]
    return diff

def warp_keypoints(keypoints, homography_mat):
    """Transform key points based on the given homography matrix"""
    # Stitch key points and a column vector with all 1 into a matrix
    source = torch.cat([keypoints, torch.ones(len(keypoints), 1).to(keypoints.device)], dim=-1)
    # Calculate the transformed target point by matrix multiplication
    dest = (homography_mat @ source.T).T
    # Normalize the dest, that is, divide the dest by the value of the third column
    # dest /= dest[:, 2:3]
    dest = dest / dest[:, 2:3]  # Update with non-in-place operations
    # Returns the first two columns of the processed target point
    return dest[:, :2]

def torch_find_matches(src_keypoints1, src_keypoints2, homography, dist_thresh=3, n_iters=1):
    '''
    Find a matching key point pair between the two sets of key points. First, the first set of key points are projected into the space of the second set of key points for comparison. 
    Then, by calculating the distance between the two sets of key points, the closest pair of key points is found. 
    Finally, based on the distance threshold and the number of iterations, the matching key point pairs are filtered and the matching key point pairs and the unmatched key point index are returned.
    '''
    # Initialize some variables and empty matching list.
    match_list_1, match_list_2 = torch.empty(0, dtype=torch.int64, device=src_keypoints1.device), torch.empty(0, dtype=torch.int64, device=src_keypoints2.device)
    # Create a list with all indexes to track unmatched key points.
    missing_indices_1 = torch.arange(len(src_keypoints1), device=src_keypoints1.device, dtype=torch.long)
    missing_indices_2 = torch.arange(len(src_keypoints2), device=src_keypoints2.device, dtype=torch.long)
    # Project the first set of key points into the space of the second set of key points.
    src_keypoints1_projected = warp_keypoints(src_keypoints1, homography) #keypoint1 must be projected to keypoint2 space for comparsion
    # For each iteration, find the closest keypoint pair and filter out the matching keypoint pair based on the distance threshold.
    for i in range(n_iters): # The second iteration can only recover 1 or 2 matching key points in most cases. So 1 is enough.
        keypoints1 = src_keypoints1_projected[missing_indices_1, :]
        keypoints2 = src_keypoints2[missing_indices_2, :]
        distance = torch_cdist(keypoints1, keypoints2)
        # if distance.size(1) == 0:
        #     raise Exception('distance is none')
        min1 = torch.argmin(distance, 1)
        min2 = torch.argmin(distance, 0)
        intersect_indexes_2 = torch.where(min1[min2] == torch.arange(len(min2), device=min1.device))[0]
        intersect_indexes1 = min2[intersect_indexes_2]
        matched_distances = distance[intersect_indexes1, intersect_indexes_2] < dist_thresh
        intersect_indexes1 = intersect_indexes1[matched_distances]
        intersect_indexes_2 = intersect_indexes_2[matched_distances]
        matched_indexes_1 = missing_indices_1[intersect_indexes1]
        matched_indexes_2 = missing_indices_2[intersect_indexes_2]
        missing_indices_1 = torch_setdiff1d(missing_indices_1, matched_indexes_1)
        missing_indices_2 = torch_setdiff1d(missing_indices_2, matched_indexes_2)
        match_list_1 = torch.cat((match_list_1, matched_indexes_1))
        match_list_2 = torch.cat((match_list_2, matched_indexes_2))
    # Returns the matching key pair and unmatched key point index.
    return match_list_1, match_list_2, missing_indices_1, missing_indices_2

def scale_homography(homo_matrix, src_height, src_width, dest_height, dest_width):
    """
    If src and warped image is scaled by same amount, then homography needs to changed according
    to the scale in x and y direction
    """
    scale_x = dest_width / src_width
    scale_y = dest_height / src_height
    scale_matrix = np.diag([scale_x, scale_y, 1.0])
    homo_matrix = scale_matrix @ homo_matrix @ np.linalg.inv(scale_matrix)
    return homo_matrix

# def resize_aspect_ratio(image, resize_h, resize_w):
#     h, w = image.shape[0:2]
#     max_size = max(h, w)
#     ratio_h, ratio_w = h / max_size, w / max_size
#     new_height, new_width = int(resize_h * ratio_h), int(resize_w * ratio_w)
#     resized = cv2.resize(image, (new_width, new_height))
#     template = np.ones((resize_h, resize_w), dtype=np.uint8) * np.random.randint(0, 127)
#     template[((resize_h - resized.shape[0])//2):((resize_h - resized.shape[0])//2) + resized.shape[0],
#               ((resize_w - resized.shape[1])//2):((resize_w - resized.shape[1])//2) + resized.shape[1]] = resized
#     return template

def resize_aspect_ratio(image, resize_h, resize_w):
    # Get the image's height, width, and number of channels (if it is a grayscale graph, the default number of channels is 1)
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    # Calculate the adjusted image size to maintain the original aspect ratio
    max_size = max(h, w)
    ratio_h, ratio_w = h / max_size, w / max_size
    new_height, new_width = int(resize_h * ratio_h), int(resize_w * ratio_w)
    # Adjust image size
    resized = cv2.resize(image, (new_width, new_height))
    # Create fill templates based on the number of channels
    if channels == 1:
        template = np.ones((resize_h, resize_w), dtype=np.uint8) * np.random.randint(0, 127)
    else:
        template = np.ones((resize_h, resize_w, channels), dtype=np.uint8) * np.random.randint(0, 127)
    # Place the adjusted image in the center of the template
    start_h, start_w = (resize_h - new_height) // 2, (resize_w - new_width) // 2
    template[start_h:start_h + new_height, start_w:start_w + new_width] = resized

    return template
