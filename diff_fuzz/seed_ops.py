import cv2
import numpy as np

import consts

# Compute tenengrad for seed filtering
def tenengrad(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    tenengrad = np.sum(grad)

    return tenengrad

# Normalization
def normalize(tenens):
    min = np.min(tenens)
    max = np.max(tenens)

    if max == min:
        return np.zeros_like(tenens)
    normed = (tenens - min) / (max - min)

    return normed

# Filter images by processing npz data
def filter_data(path):
    with np.load(path) as f:
        imgs = f['advs']

    res = []
    tenen_values = []

    for img in imgs:
        # should be gray scale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        val = tenengrad(img)
        tenen_values.append(val)

    norm_val = normalize(tenen_values)

    filtered_idxs = np.where(norm_val >= consts.CLARITY_THRESHOLD)[0]
    filtered_advs = imgs[filtered_idxs]

    np.savez(consts.FILTER_SAMPLE_PATH, advf = filtered_advs)
      
