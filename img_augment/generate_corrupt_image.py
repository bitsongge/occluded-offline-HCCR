import cv2
import random
import numpy as np
from glob import glob


PATH = glob('/home/alyssa/PythonProjects/occluded/dataset/trainset/amplicationset/*')
TARGET_PATH = "/Users/albert/con_lab/alyssa/OCCLUDED/D11/"


def diffRegions(flag):
    count = 0
    for path in PATH:
        name = glob(path+'/*')
        for img_name in name:
            img_ori = cv2.imread(img_name)
            img = img_ori.copy()
            [rows, cols, dims] = img.shape

            box_width = int((cols*rows*0.25)**0.5)
            box_height = box_width
            mask = np.ones((rows, cols, dims))*255

            if(flag == "top"):
                x = 0
                y = int(cols/2 - box_height/2)
            elif(flag == "medium"):
                x = int(rows/2 - box_width/2)
                y = int(cols/2 - box_height/2)
            elif(flag == "bottom"):
                x = int(cols - box_height)
                y = int(rows/2 - box_width/2)

            for i in range(x, x+box_width):
                for j in range(y, y+box_height):
                    img[i, j, :] = 0
                    mask[i, j, :] = 0

            img_final = np.concatenate((img_ori, img, mask), axis=1)
            cv2.imwrite(TARGET_PATH + str(count) + ".png", img_final)
            count += 1


def generateRectangular(scale, num):
    outer_iter = 0
    inner_iter = 0
    for path in PATH:
        outer_iter += 1
        name = glob(path+'/*')
        for img_name in name:
            inner_iter += 1
            if inner_iter == 60:
                inner_iter = 0
                break
            img_ori = cv2.imread(img_name)
            img = img_ori.copy()
            [rows, cols, dims] = img.shape

            box_width = int((cols*rows*scale)**0.5)
            box_height = box_width
            mask = np.ones((rows, cols, dims))*255

            for k in range(num):
                x = random.randint(0, int(rows-box_height))
                y = random.randint(0, int(cols-box_width))
                for i in range(x, x+box_height):
                    for j in range(y, y+box_width):
                        img[i, j, :] = 0
                        mask[i, j, :] = 0

            img_final = np.concatenate((img_ori, img, mask), axis=1)
            cv2.imwrite(TARGET_PATH + str(outer_iter) + "_" + str(inner_iter) + ".png", img_final)


def generateRound(scale, num):
    outer_iter = 0
    inner_iter = 0
    for path in PATH:
        outer_iter += 1
        name = glob(path+'/*')
        for img_name in name:
            inner_iter += 1
            if inner_iter == 60:
                inner_iter = 0
                break
            img_ori = cv2.imread(img_name)
            img = img_ori.copy()
            [rows, cols, dims] = img.shape
            
            round_area = rows*cols*scale
            radius = int((round_area/3.14)**0.5)
            mask = np.ones((rows, cols, dims))*255

            for k in range(num):
                x = random.randint(int(0+radius), int(rows-radius))
                y = random.randint(int(0+radius), int(cols-radius))
                for i in range(x-radius, x+radius):
                    for j in range(y-radius, y+radius):
                        lx = abs(i - x)
                        ly = abs(j - y)
                        l = (pow(lx, 2) + pow(ly, 2)) ** 0.5
                        if(l < radius):
                            img[i, j, :] = 0
                            mask[i, j, :] = 0

            img_final = np.concatenate((img_ori, img, mask), axis=1)
            cv2.imwrite(TARGET_PATH + str(outer_iter) + "_" + str(inner_iter) + ".png", img_final)


def removePixels(scale):
    outer_iter = 0
    inner_iter = 0
    for path in PATH:
        outer_iter += 1
        name = glob(path+'/*')
        for img_name in name:
            inner_iter += 1
            if inner_iter == 60:
                inner_iter = 0
                break
            img_ori = cv2.imread(img_name)
            img = img_ori.copy()
            [rows, cols, dims] = img.shape
            
            round_area = int(rows*cols*scale)
            mask = np.ones((rows, cols, dims))*255

            for k in range(round_area):
                x = random.randint(0, int(rows))
                y = random.randint(0, int(cols))
                for i in range(0, int(rows)):
                    for j in range(0, int(cols)):
                        if(i == x and j == y):
                            img[i, j, :] = 0
                            mask[i, j, :] = 0

            img_final = np.concatenate((img_ori, img, mask), axis=1)
            cv2.imwrite(TARGET_PATH + str(outer_iter) + "_" + str(inner_iter) + ".png", img_final)


if __name__ == "__main__":
    scale = 0.2
    num = 1

    # generateRectangular(scale, num)
    # generateRound(scale, num)
    # removePixels(scale)
    # diffRegions("bottom")
