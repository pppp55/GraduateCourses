import os.path as osp

import cv2
import numpy as np


def histogram_equalization(img):
    """Returns the image after histogram equalization.
    Args:
        img: the input image to be executed for histogram equalization.
    Returns:
        res_img: the output image after histogram equalization.
    """
    if img is None:
        raise ValueError("img is None")
    if img.ndim != 2:
        raise ValueError("img must be a grayscale (2D) image")

    src = img.astype(np.uint8)
    hist = np.bincount(src.ravel(), minlength=256)
    cdf = np.cumsum(hist)

    non_zero = np.nonzero(hist)[0]
    if non_zero.size <= 1:
        return src.copy()

    cdf_min = cdf[non_zero[0]]
    denom = src.size - cdf_min
    if denom <= 0:
        return src.copy()

    lut = np.round((cdf - cdf_min) * 255.0 / denom)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    res_img = lut[src]

    return res_img


def local_histogram_equalization(img):
    """Returns the image after local histogram equalization.
    Args:
        img: the input image to be executed for local histogram equalization.
    Returns:
        res_img: the output image after local histogram equalization.
    """
    if img is None:
        raise ValueError("img is None")
    if img.ndim != 2:
        raise ValueError("img must be a grayscale (2D) image")

    src = img.astype(np.uint8)
    h, w = src.shape

    # Square neighborhood size (odd): tune this to control local contrast strength.
    window_size = 31
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size must be an odd integer >= 3")
    radius = window_size // 2

    padded = np.pad(src, ((radius, radius), (radius, radius)), mode="reflect")
    res_img = np.zeros_like(src)
    total = window_size * window_size

    for i in range(h):
        row_start = i
        row_end = i + window_size

        block = padded[row_start:row_end, 0:window_size]
        hist = np.bincount(block.ravel(), minlength=256).astype(np.int32)

        for j in range(w):
            val = src[i, j]
            cdf = np.cumsum(hist)
            non_zero = np.nonzero(hist)[0]

            if non_zero.size <= 1:
                res_img[i, j] = val
            else:
                cdf_min = cdf[non_zero[0]]
                denom = total - cdf_min
                if denom <= 0:
                    res_img[i, j] = val
                else:
                    mapped = np.round((cdf[val] - cdf_min) * 255.0 / denom)
                    res_img[i, j] = np.uint8(np.clip(mapped, 0, 255))

            if j < w - 1:
                old_col = padded[row_start:row_end, j]
                new_col = padded[row_start:row_end, j + window_size]
                hist -= np.bincount(old_col, minlength=256)
                hist += np.bincount(new_col, minlength=256)

    return res_img


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "moon.png"), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res_hist_equalization = histogram_equalization(img)
    res_local_hist_equalization = local_histogram_equalization(img)

    cv2.imwrite(osp.join(root_dir, "HistEqualization.jpg"), res_hist_equalization)
    cv2.imwrite(
        osp.join(root_dir, "LocalHistEqualization.jpg"), res_local_hist_equalization
    )
