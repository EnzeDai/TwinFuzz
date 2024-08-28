import cv2
import numpy as np

import consts

# Compute tenengrad for seed filtering
def tenengrad(img):

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    tenengrad = np.sum(grad)

    return tenengrad

# Normalization
# def normalize(tenens):
#     min = np.min(tenens)
#     max = np.max(tenens)

#     if max == min:
#         return np.zeros_like(tenens)
#     normed = (tenens - min) / (max - min)

#     return normed

# Filter images by processing npz data
def filter_data(path):
    with np.load(path) as f:
        imgs = f['advs']

    filtered_advs = []
    tenen_values = []

    for img in imgs:
        # should be gray scale
        if img.shape[2] != 1:
            raise ValueError("Image should be gray scale.")
        
        val = tenengrad(img)
        tenen_values.append(val)

    threshold = np.full_like(tenen_values, consts.CLARRITY_THRESHOLD)

    filtered_idxs = np.where(tenen_values < threshold)[0]
    filtered_advs = imgs[filtered_idxs]

    np.savez(consts.FILTER_SAMPLE_PATH, advf = filtered_advs)

    return filtered_idxs
      
