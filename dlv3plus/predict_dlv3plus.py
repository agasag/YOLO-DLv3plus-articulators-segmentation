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


###

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        #image = image / 127.5 - 1
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    """mask = tf.one_hot(tf.cast(read_image(mask_list, mask=True), tf.uint8), depth=4)
    mask_shape = tf.shape(mask)
    mask = tf.reshape(mask, [mask_shape[0], mask_shape[1], mask_shape[3]])
    print(mask)"""
    return image, mask

def infer(model, image_tensor):
    import time
    t1 = time.time()
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    t2 = time.time()
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions, t1, t2


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()


def plot_predictions(images_list, colormap, model):
    prediction_masks_all = []
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )
        prediction_masks_all.append(prediction_mask)
    return prediction_masks_all

def predictions_test(images_list, colormap, model):
    prediction_masks_all = []
    t1_all = []
    t2_all = []
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask, t1, t2  = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        prediction_masks_all.append(prediction_mask)
        t1_all.append(t1)
        t2_all.append(t2)
    return prediction_masks_all, t1_all, t2_all

print('Model loading...')
model = keras.models.load_model('.\\14-03-2023-00-09-06_xception_2\\')
print('Model loaded...')

colormap = loadmat(
    "./human_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)

print('Get data from patients...')
retrieve_all_patients = '.\\images\\'
subdirs = [item for item in os.listdir(retrieve_all_patients) if os.path.isdir(os.path.join(retrieve_all_patients, item)) ]
subdirs_temp = subdirs
subdirs = subdirs

print('Data processing...')
for patients in subdirs:

    all_patients_vids = [x for x in os.walk(retrieve_all_patients + '\\' + patients)]
    all_patients_vids = all_patients_vids[0]
    root_folder_patient = all_patients_vids[0]
    subdirs_patient = all_patients_vids[1]

    for vids in subdirs_patient:

        DATA_DIR_path_r = '.\\images\\' + patients + '\\' + vids + '\\right\\images_roi\\'
        DATA_DIR_path_l = '.\\images\\' + patients + '\\' + vids + '\\left\\images_roi\\'

        path_r = '.\\images\\' + patients + '\\' + vids + '\\right\\segmentation\\'
        path_l = '.\\images\\' + patients + '\\' + vids + '\\left\\segmentation\\'

        isExistr = os.path.exists(path_r)
        isExistl = os.path.exists(path_l)

        if isExistr is False:
            os.makedirs(path_r)
        if isExistl is False:
            os.makedirs(path_l)

        images_camera_r = sorted(glob(DATA_DIR_path_r + '*'))
        images_camera_l = sorted(glob(DATA_DIR_path_l + '*'))

        preds_r, t1_r, t2_r = predictions_test(images_camera_r, colormap, model=model)
        preds_l, t1_l, t2_l = predictions_test(images_camera_l, colormap, model=model)

        print('Video: ' + vids + ', right')
        list_of_imgs_r = []
        for preds_idx_r in range(0, len(preds_r)):
            fil_name = images_camera_r[preds_idx_r].split('\\')
            list_of_imgs_r.append(preds_r[preds_idx_r])
            cv2.imwrite(path_r + fil_name[-1],
                        preds_r[preds_idx_r])

        print('Video: ' + vids + ', left')
        list_of_imgs_l = []
        for preds_idx_l in range(0, len(preds_l)):
            fil_name = images_camera_l[preds_idx_l].split('\\')
            list_of_imgs_l.append(preds_l[preds_idx_l])
            cv2.imwrite(path_l + fil_name[-1],
                        preds_l[preds_idx_l])
