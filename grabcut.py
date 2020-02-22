import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from math import ceil

input_folder = 'train/YL_Tomato/'
input_file = 'dataset.txt'
output_folder = 'out/YL_Tomato/'
img_size = 200

def grabcut_genmask(image, iter):
    IMG_H, IMG_W = image.shape[:2]
    image = cv.resize(image, (IMG_W, IMG_H))
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, IMG_W, IMG_H)
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, iter, cv.GC_INIT_WITH_RECT)
    #mark2 is 0 with cv.GC_PR_BGD and cv.GC_BGD
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

if __name__ == "__main__":
    with open(input_folder+input_file, 'r') as f:
        train_data = f.read()
        train_data = train_data.split('\n')

    for img_file in train_data:
        img = cv.imread(input_folder+img_file)
        #cv.imshow('raw', img)
        #cv.waitKey(1)
        img_X = int(img_size)
        img_Y = int(img.shape[0]*(img_X/img.shape[1]))

        imgresize = cv.resize(img, (img_X, img_Y))
        mask = grabcut_genmask(imgresize, 6)
        imgcut = imgresize * mask[:, :, np.newaxis]

        #cv.imshow('imgcut', imgcut)

        cv.imwrite(output_folder + img_file, imgcut)
        print(img_file)

    #cv.imshow('i', img)
    #cv.imwrite(output_folder + '1.jpg', img)
    #cv.waitKey(0)