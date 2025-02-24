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
from loss import dice_coef, jacard, dice_coef_loss, iou_loss, tversky, tversky_loss, focal_tversky, generalized_dice_coeff, generalized_dice_loss
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

def augment_using_ops(images, labels):

	return (images, labels)

def data_generator(image_list, mask_list):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomContrast(0.1),
        # tf.keras.layers.RandomCrop(height=150, width=150),
        tf.keras.layers.RandomFlip(mode='horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        # tf.keras.layers.RandomHeight(0.5),
        # tf.keras.layers.RandomWidth(0.1),
        #tf.keras.layers.experimental.preprocessing.RandomBrightness(0.2),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ])

    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    #dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

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

def dice_coef(y_true, y_pred, smooth = 0.01):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel_loss(y_true, y_pred, numLabels=4):
    dice=0
    print('ok')
    print(y_true)
    print(y_pred)
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,index,:,:,:], y_pred[:,index,:,:,:])
    return 1 - dice

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

def DICE_COE(pred, gt):
    """intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading"""
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        dice = np.sum(pred[gt == 1]) * 2.0 / (
                np.sum(pred) + np.sum(gt))

    if np.sum(pred[gt == 1]) == 0 and (np.sum(pred) + np.sum(gt))>0:
        dice = 0

    if np.sum(pred[gt == 1]) > 0 and (np.sum(pred) + np.sum(gt))>0:
        dice = np.sum(pred[gt == 1]) * 2.0 / (
                np.sum(pred) + np.sum(gt))
    return dice

def conf_mat_jacard(Y_pred, Y_val):
    FP = len(np.where(Y_pred - Y_val  == 1)[0])
    FN = len(np.where(Y_pred - Y_val  == -1)[0])
    TP = len(np.where(Y_pred + Y_val ==2)[0])
    TN = len(np.where(Y_pred + Y_val == 0)[0])
    cmat = [[TP, FN], [FP, TN]]
    weight = len(np.where(Y_val  == 1)[0])

    try:
        jacard_ind = TP/(TP+FP+FN)
    except:
        jacard_ind = 0
    return cmat, jacard_ind, weight

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

def generalized_dice_coeff(y_true, y_pred):
    print(y_pred)
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)

    gen_dice_coef = 2*numerator/denominator

    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

class MeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
        super(MeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

""" PARAMS TO SET """
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
mode = "scratch" # either "scratch" or "fine-tune"

""" train - val """
DATA_DIR = ".\\train\\"
folder_path = Path(DATA_DIR)
images_len = sum(1 for file in folder_path.iterdir() if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"})

NUM_TRAIN_IMAGES = int(round(0.8*images_len))
NUM_VAL_IMAGES = images_len-NUM_TRAIN_IMAGES

train_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "labels/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "labels/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]

train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

""" test (ground truth) """
DATA_DIR2 = ".\\GTruth\\"
test_images = sorted(glob(os.path.join(DATA_DIR2, "images/*")))
test_masks = sorted(glob(os.path.join(DATA_DIR2, "labels/*")))
test_dataset = data_generator(test_images, test_masks)

""" print summary """
print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

DATA_DIR3= '.\\images\\'
test_images2 = sorted(glob(os.path.join(DATA_DIR3, "images/*")))

if mode is "scratch":
    model = Deeplabv3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), classes=NUM_CLASSES, backbone='xception')
elif mode is "fine-tune":
    model = keras.models.load_model('.\\14-03-2023-00-09-06_xception_2\\')

model.summary()

""" training hyperparameters """
reduceLr = keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.3, monitor='val_sparse_categorical_accuracy')
early_stopping = keras.callbacks.EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss= loss,
    metrics=["sparse_categorical_accuracy",
             "accuracy"],
)

""" train """
print('Training process started...')
history = model.fit(train_dataset, batch_size=32, validation_data=val_dataset, epochs=150, callbacks=[reduceLr, early_stopping])

now = datetime.now()
model.save('./' + now.strftime("%d-%m-%Y-%H-%M-%S") + '_xception_backbone')
plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()

# Loading the Colormap
colormap = loadmat(
    "./human_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


preds, t1, t2 = predictions_test(test_images, colormap, model=model)
