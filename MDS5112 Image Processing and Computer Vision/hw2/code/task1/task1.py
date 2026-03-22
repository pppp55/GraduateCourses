import os.path as osp

import cv2
import numpy as np


def gaussian_filter(img, kernel_size, sigma):
    """Returns the image after Gaussian filter.
    Args:
        img: the input image to be Gaussian filtered.
        kernel_size: the kernel size in both the X and Y directions.
        sigma: the standard deviation in both the X and Y directions.
    Returns:
        res_img: the output image after Gaussian filter.
    """
    if img is None:
        raise ValueError("img is None")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than 1")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    # sigma=0 => output is the same as input
    if sigma == 0:
        return img.copy()

    radius = kernel_size // 2

    # generate a 2D Gaussian kernel and normalize it
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel /= np.sum(kernel)

    img_float = img.astype(np.float64)

    if img_float.ndim == 2:
        padded = np.pad(img_float, ((radius, radius), (radius, radius)), mode="reflect")
        res_img = np.zeros_like(img_float)

        h, w = img_float.shape
        for i in range(h):
            for j in range(w):
                patch = padded[i : i + kernel_size, j : j + kernel_size]
                res_img[i, j] = np.sum(patch * kernel)
    elif img_float.ndim == 3:
        padded = np.pad(
            img_float,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="reflect",
        )
        res_img = np.zeros_like(img_float)

        h, w, c = img_float.shape
        for i in range(h):
            for j in range(w):
                patch = padded[i : i + kernel_size, j : j + kernel_size, :]
                for ch in range(c):
                    res_img[i, j, ch] = np.sum(patch[:, :, ch] * kernel)
    else:
        raise ValueError("img must be a 2D or 3D numpy array")

    # back to the original image type range
    res_img = np.clip(res_img, 0, 255).astype(img.dtype)
    return res_img


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "Lena-RGB.jpg"))
    kernel_size = 5
    sigma = 1
    res_img = gaussian_filter(img, kernel_size, sigma)

    cv2.imwrite(osp.join(root_dir, "gaussian_result.jpg"), res_img)
