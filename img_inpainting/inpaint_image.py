import numpy as np
from glob import glob
import cv2
import skimage.measure
from model import generator_model


def load_data(start, end):
    path = glob('/home/alyssa/PythonProjects/occluded/dataset/test_D15/*')
    batch_images = path[start:end]

    imgs_hr = []
    imgs_lr = []
    for img_path in batch_images:
        img = cv2.imread(img_path)
        w = 64

        img_hr = img[:, 0:w]
        img_lr = img[:, w:2*w]

        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)

    imgs_hr = np.array(imgs_hr) / 127.5 - 1
    imgs_lr = np.array(imgs_lr) / 127.5 - 1

    return imgs_hr, imgs_lr


def inpaint():
    g = generator_model()
    g.load_weights('/home/alyssa/PythonProjects/occluded/key_code/img_inpainting/weights/D17/generator_80000_47.h5')

    sum_ssim = 0
    sum_ac = 0
    count = 0
    for i in range(9440):
        y_pre, x_pre = load_data(i*4, (i+1)*4)

        generated_images = g.predict(x=x_pre, batch_size=2)
        result = (generated_images+1)*127.5
        re = (x_pre+1)*127.5
        ori = (y_pre+1)*127.5

        for j in range(4):
            count += 1
            cv2.imwrite('/home/alyssa/PythonProjects/occluded/test_image/D17/'+str(count)+'_0.jpg', ori[j])
            cv2.imwrite('/home/alyssa/PythonProjects/occluded/test_image/D17/'+str(count)+'_1.jpg', result[j])
            cv2.imwrite('/home/alyssa/PythonProjects/occluded/test_image/D17/'+str(count)+'_2.jpg', re[j])

            a = cv2.imread('/home/alyssa/PythonProjects/occluded/test_image/D17/'+str(count)+'_0.jpg')
            b = cv2.imread('/home/alyssa/PythonProjects/occluded/test_image/D17/'+str(count)+'_1.jpg')
            c = cv2.imread('/home/alyssa/PythonProjects/occluded/test_image/D17/'+str(count)+'_2.jpg')

            # ab = skimage.measure.compare_psnr(a, b)
            # ac = skimage.measure.compare_psnr(a, c)

            ab_ssim = skimage.measure.compare_ssim(a, b, multichannel=True)
            ac_ssim = skimage.measure.compare_ssim(a, c, multichannel=True)

            print(count, " ", ab_ssim)

            sum_ssim = sum_ssim + ab_ssim
            sum_ac = sum_ac + ac_ssim

    percent = sum_ssim / count
    percent_ac = sum_ac / count
    print("result:", percent, " ", percent_ac)


inpaint()
