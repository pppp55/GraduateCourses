import numpy as np
import cv2

def image_t(im, scale=1.0, rot=45, trans=(50,-50)):
    # TODO Write "image affine transformation" function based on the illustration in specification.
    # Return transformed result image
    if im is None:
        raise ValueError('Input image is None. Please check the image path.')

    h, w = im.shape[:2]
    tx, ty = trans

    # Sample three points from the source image (a triangle).
    src_pts = np.array(
        [
            [0.0, 0.0],
            [float(w - 1), 0.0],
            [0.0, float(h - 1)],
        ],
        dtype=np.float32,
    )

    # Rotate/scale around the image center, then translate.
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    theta = np.deg2rad(rot)

    # Note: image coordinate system has y-axis pointing down.
    # To make a positive angle visually anticlockwise, we use -theta here.
    angle = -theta
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))

    dst_pts = []
    for x, y in src_pts:
        dx = float(x) - cx
        dy = float(y) - cy
        x2 = scale * (cos_a * dx - sin_a * dy) + cx + tx
        y2 = scale * (sin_a * dx + cos_a * dy) + cy + ty
        dst_pts.append([x2, y2])
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Compute affine transform and warp.
    M = cv2.getAffineTransform(src_pts, dst_pts)
    result = cv2.warpAffine(
        im,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return result


if __name__ == '__main__':
    im = cv2.imread('./misc/pearl.jpeg')
    
    scale  = 0.5
    rot    = 45
    trans  = (50, -50)
    result = image_t(im, scale, rot, trans)
    cv2.imwrite('./results/affine_result.png', result)