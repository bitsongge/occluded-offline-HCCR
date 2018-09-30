import numpy as np
from glob import glob
import cv2
from model import generator_model


def load_data(start, end):
    path = glob('/home/alyssa/Desktop/the_second/D5/*')
    batch_images = path[start:end]

    imgs_hr = []
    for img_path in batch_images:
        img = cv2.imread(img_path)
        imgs_hr.append(img)

    imgs_hr = np.array(imgs_hr) / 127.5 - 1
    return imgs_hr


def deblur_real():
    g = generator_model()
    g.load_weights('/Users/albert/con_lab/alyssa/OCCLUDED/generator_57000_243.h5')

    count = 0
    for i in range(10):
        x_pre = load_data(i*2, (i+1)*2)
        generated_images = g.predict(x=x_pre, batch_size=1)
        result = (generated_images+1)*127.5

        for j in range(2):
            count += 1
            cv2.imwrite('/home/alyssa/PythonProjects/occluded/11/'+str(count)+'_1.jpg', result[j])

 
deblur_real()
