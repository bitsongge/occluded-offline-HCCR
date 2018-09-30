import numpy as np
import cv2
from glob import glob

RESHAPE = (64, 64)


def load_data(batch_size=1):
    path = glob('/Users/albert/con_lab/alyssa/OCCLUDED/D333_bottom/*')
    batch_images = np.random.choice(path, size=batch_size)

    imgs_hr = []
    imgs_lr = []
    masks = []
    for img_path in batch_images:
        img = cv2.imread(img_path)
        w = 64

        img_hr = img[:, 0:w]
        img_lr = img[:, w:2*w]
        mask = img[:, 2*w:]

        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)
        masks.append(mask)

    imgs_hr = np.array(imgs_hr) / 127.5 - 1
    imgs_lr = np.array(imgs_lr) / 127.5 - 1
    masks = np.array(masks) / 255

    return imgs_hr, imgs_lr, masks
