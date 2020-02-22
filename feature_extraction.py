import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd

# Read image data
# Red tomato
is_training = False

classes = {'brown': 0, 'orange': 1, 'yellow': 2}

if is_training:
    data_folder = 'training\\'
else:
    data_folder = 'testing\\'



# plt.figure()
# list_files1 = glob.glob(data_folder + 'brown' +'/' + '*.jpg')
# list_files2 = glob.glob(data_folder + 'orange' +'/' + '*.jpg')
# list_files3 = glob.glob(data_folder + 'Yellow' + '/' + '*.jpg')
# img = cv2.imread(list_files1[0], 0)
# plt.hist(img.ravel(),256,[0,256])
# plt.figure()
# img = cv2.imread(list_files2[0], 0)
# plt.hist(img.ravel(),256,[0,256])
# plt.figure()
# img = cv2.imread(list_files3[0], 0)
# plt.hist(img.ravel(),256,[0,256]); plt.show()


def his_extract(img):
    # extract histogram on H channel and S channel
    his_H = cv2.calcHist([img], [0], None, [256], [0, 256])
    his_S = cv2.calcHist([img], [1], None, [256], [0, 256])

    his = np.concatenate((his_H, his_S))

    # Normalize histogram
    his = np.true_divide(his, img.shape[0]*img.shape[1]*2)
    return his


if __name__ =='__main__':
    dataset_labels =[]
    for id_cls, cls in enumerate(list(classes.keys())):
        cls_folder = cls + '\\'
        list_files = glob.glob(data_folder + cls_folder + '*.jpg')
        data = []
        for img_id, im_pth in enumerate(list_files):
            # write to label file
            image = cv2.imread(im_pth)
            img_name = im_pth.split('\\')[-2] + '_' + im_pth.split('\\')[-1].split('.')[0]

            # Apply Gaussian to remove noise
            image_blr = cv2.blur(image, (5, 5))

            # Change to HSV color and extract histogram feature
            image_hsv = cv2.cvtColor(image_blr, cv2.COLOR_BGR2HSV)
            histogram = his_extract(image_hsv)

            # store to file
            feature_pth = data_folder + 'feature\\'
            #label_pth = data_folder + 'label\\'
            if not os.path.exists(feature_pth):
                os.makedirs(feature_pth)
            #if not os.path.exists(label_pth):
            #    os.makedirs(label_pth)

            file_pth = img_name + '.npy'
            store_pth = feature_pth + file_pth
            np.save(store_pth, histogram)

            dataset_labels.append([file_pth, str(classes[cls])])
            # if(id_cls == 0 and img_id == 0 ):
            #     csvfile = open(feature_pth + 'label.csv', 'w', newline='')
            # else:
            #     csvfile = open(feature_pth + 'label.csv', 'a', newline='')
            # csvwriter = csv.writer(csvfile, delimiter=' ')
            # csvwriter.writerow([file_pth, str(classes[cls])])
            # csvfile.close()
    pd.DataFrame(dataset_labels).to_csv(feature_pth + 'label.csv')

