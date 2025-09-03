import numpy as np
import glob
from PIL import Image


def min_max_percentile(image: np.ndarray, pmin: float = 0.5, pmax: float = 99.5):
    vmin = np.percentile(image, pmin)
    vmax = np.percentile(image, pmax)
    image = (image - vmin) / (vmax - vmin)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def split_mosaic(img: np.ndarray, rows: int, cols: int):
    if img.ndim == 2:
        img = img[None, ...]  # 轉成 (1, H, W)

    N, H_total, W_total = img.shape
    H_patch = H_total // rows
    W_patch = W_total // cols

    patches = []

    for frame in img:
        for i in range(rows):
            for j in range(cols):
                patch = frame[
                    i * H_patch : (i + 1) * H_patch,
                    j * W_patch : (j + 1) * W_patch
                ]
                patches.append(patch)

    return np.stack(patches)


def save_mosaic(patch_dir, raw_images=None, n_cols=3, auto_contrast=False):
    files = sorted(glob.glob("%s/[!m]*.tif"%patch_dir))
    imgs = []

    if raw_images is not None:
        imgs = raw_images
    else:
        for f in files:
            imgs.append(np.array(Image.open(f)))
        imgs = np.array(imgs)

    size = imgs.shape[1]
    n_rows = len(imgs)//n_cols
    mosaic_auto = np.zeros((n_rows*size, n_cols*size))
    mosaic_orig = np.zeros((n_rows*size, n_cols*size))
    img_id = 0
    for i in reversed(range(n_rows)):
        for j in range(n_cols):
            im = imgs[img_id]

            # no auto contrast
            mosaic_orig[i*size:(i+1)*size, j*size:(j+1)*size] = im

            if auto_contrast:
                if i!=(n_rows-1) or j!=0:
                    diff = []
                    if i==n_rows-1 or j>0:
                        b_avg = np.mean(im[:, :50])
                        prev_b_avg = np.mean(imgs[img_id-1][:, -50:])
                        diff.append(prev_b_avg/b_avg)
                    if i<(n_rows-1):
                        b_avg = np.mean(im[-50:, :])
                        prev_b_avg = np.mean(imgs[img_id-n_cols][:50, :])
                        diff.append(prev_b_avg/b_avg)
                    im = im*np.mean(diff)
                    imgs[img_id] = im
                mosaic_auto[i*size:(i+1)*size, j*size:(j+1)*size] = im

            img_id += 1   

    if raw_images is not None:
        save_path = f'{patch_dir}/mosaic_from_raw.tif'
    else:
        save_path = f'{patch_dir}/mosaic_processed.tif'

    mosaic_orig = min_max_percentile(mosaic_orig, 0, 100)
    Image.fromarray(mosaic_orig).save(save_path)

    if auto_contrast:
        save_path = f'{patch_dir}/mosaic_processed(auto).tif'
        mosaic_auto = min_max_percentile(mosaic_auto, 0, 100)
        Image.fromarray(mosaic_auto).save(save_path)