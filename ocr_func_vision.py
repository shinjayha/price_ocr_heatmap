import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


""" load image """
def _list_files(dir='./input/'):
    import os
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files



# type result and draw bbox over image
def _img_ocr_result(img, ocr_result, FONT_SIZE=10):
    import cv2
    for _, row in ocr_result.iterrows():
        x, y, w, h = row["left"], row["top"], row["width"], row["height"]
        plt.text(x, (y - 10), row["text"], fontsize=FONT_SIZE, color="red")
        cv2.rectangle(img,(int(x), int(y)),(int(x) + int(w), int(y) + int(h)),(255, 0, 0),1 )
    return img


# K-means Clustering
def _kmeanclustered(img, KNUM=2):
    img_2 = img.copy().reshape(-1, 3)
    img_2 = np.float32(img_2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = KNUM
    ret, label, center = cv2.kmeans(img_2, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS )
    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((img.shape))
    return img


# radial heatmap
def _heatmap_2(img, ocr_result):
    from scipy.interpolate.rbf import Rbf  # radial basis functions
    if len(ocr_result) <= 2:
        # avoid [ValueError: zero-size array to reduction operation maximum which has no identity]
        ocr_result = pd.DataFrame(data=[], columns=["left", "top", "width", "height", "conf", "text"])
        ocr_result.loc[len(ocr_result)] = [0, 0, 0, 0, 0.0, "1"]
        ocr_result.loc[len(ocr_result)] = [0, 1, 0, 0, 0.0, "1"]
    x = ocr_result["left"] + (ocr_result["width"] // 2)
    y = ocr_result["top"] + (ocr_result["height"] // 2)
    z = ocr_result["text"].astype(np.int64)
    # https://stackoverflow.com/questions/51647590/2d-probability-distribution-with-rbf-and-scipy
    rbf_adj = Rbf(x, y, z, function="gaussian")
    dh, dw, _ = img.shape
    x_fine = np.linspace(0, dw, num=81)  # (start, stop, step)
    y_fine = np.linspace(0, dh, num=82)
    x_grid, y_grid = np.meshgrid(x_fine, y_fine)
    z_grid = rbf_adj(x_grid.ravel(), y_grid.ravel()).reshape(x_grid.shape)
    return x_fine, y_fine, z_grid



def _ocr_result_process(ocr_result, UPPER_LIMIT, CONFIDENCE=0.0):
    if len(ocr_result) != 0:
        ocr_result = ocr_result.dropna(axis=0).reset_index(drop=True)
        ocr_result["text"] = ocr_result["text"].astype("string")
        for idx, str_item in enumerate(ocr_result["text"]):
            for char in str_item:
                if not char.isdigit():
                    str_item = str_item.replace(char, "")
                    ocr_result.loc[
                        idx, "text"
                    ] = str_item  # remove non-numeric char from DataFrame, such as comma
                if len(str_item) < 3:
                    ocr_result.loc[idx, "text"] = ""
                if len(str_item) > 9:
                    ocr_result.loc[idx, "text"] = ""
        ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)
        for idx, str_item in enumerate(ocr_result["text"]):
            if ocr_result.loc[idx, "conf"] < CONFIDENCE:  # threshold
                ocr_result.loc[idx, "text"] = ""
                continue
            if int(str_item) > UPPER_LIMIT:
                ocr_result.loc[idx, "text"] = ""
                continue
            if int(str_item) < 1000:                    # option for KRW (change if USD or other)
                ocr_result.loc[idx, "text"] = ""
                continue
            if (str_item[-2] + str_item[-1]) != "00":
                ocr_result.loc[idx, "text"] = ""
        ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)
        # ocr_result.loc[len(ocr_result)] = [0, 0, 0, 0, 0.0, "0"]   # "scipy.interpolate.rbf" if all results were the same, then there is nothing to interpolate, so add an extra value
    return ocr_result
