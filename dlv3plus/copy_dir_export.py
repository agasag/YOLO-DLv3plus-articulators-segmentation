from model import Deeplabv3
import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from datetime import datetime
from sklearn.metrics import jaccard_score, accuracy_score
#from loss import dice_coef, jacard, dice_coef_loss, iou_loss, tversky, tversky_loss, focal_tversky, generalized_dice_coeff, generalized_dice_loss
from tensorflow.python.keras.optimizer_v2.adam import Adam
from distutils.dir_util import copy_tree

print('Get data...')
retrieve_all_patients = 'D:\\AgataS\\!cechy\\!images\\'
subdirs = [item for item in os.listdir(retrieve_all_patients) if os.path.isdir(os.path.join(retrieve_all_patients, item)) ]
subdirs_temp = subdirs
subdirs = subdirs
"""all_folders = all_folders[0]
root_folder = all_folders[0]
subdirs = all_folders[1]"""

print('Data processing...')
for patients in subdirs:

    all_patients_vids = [x for x in os.walk(retrieve_all_patients + '\\' + patients)]
    all_patients_vids = all_patients_vids[0]
    root_folder_patient = all_patients_vids[0]
    subdirs_patient = all_patients_vids[1]

    for vids in subdirs_patient:

        DATA_DIR_path_r = 'D:\\AgataS\\!cechy\\!images\\' + patients + '\\' + vids + '\\right\\images_roi_segm\\'
        DATA_DIR_path_l = 'D:\\AgataS\\!cechy\\!images\\' + patients + '\\' + vids + '\\left\\images_roi_segm\\'

        path_r = 'D:\\AgataS\\!cechy\\!export\\' + patients + '\\' + vids + '\\right\\images_roi_segm\\'
        path_l = 'D:\\AgataS\\!cechy\\!export\\' + patients + '\\' + vids + '\\left\\images_roi_segm\\'

        isExistr = os.path.exists(path_r)
        isExistl = os.path.exists(path_l)

        if isExistr is False:
            os.makedirs(path_r)
        if isExistl is False:
            os.makedirs(path_l)

        print('Video: ' + vids + ', right')
        copy_tree(DATA_DIR_path_r, path_r)

        print('Video: ' + vids + ', left')
        copy_tree(DATA_DIR_path_l, path_l)
