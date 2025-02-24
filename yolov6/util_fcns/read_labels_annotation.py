import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def class_to_color(class_id):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return colors[class_id]


# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, annotation):
    if annotation.isnull().values.any():
        return

    x_min, y_min = int(annotation['x_min']), int(annotation['y_min'])
    x_max, y_max = int(annotation['x_max']), int(annotation['y_max'])

    class_id = int(annotation['class_id'])
    color = class_to_color(class_id)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)


# draw all annotation bounding boxes on an image
def annotate_image(img, name, all_annotations):
    annotations = all_annotations[all_annotations['image_id'] == name]
    for index, row in annotations.iterrows():
        draw_bounding_box(img, row)


all_yaml = glob.glob('F:\\!cechy\\!yaml\\*.yaml')

for yamls in all_yaml:

    print('Vid' + yamls.split('\\')[-1][:-5] + ' processing...')
    df = pd.read_json('D:\\!cechy\\YOLOv6\\tools\\runs\\val\\' + yamls.split('\\')[-1][:-5] +'\\predictions.json')

    for idx in range(0, len(df)):
        print('- ' + yamls.split('\\')[-1][:-5] + ': ' + str(idx))
        if df.category_id[idx] == 0:
            b_box = np.round(df.bbox[idx]).astype(int)
            im = cv2.imread('F:\\!cechy\\!images\\' + yamls.split('\\')[-1].split('-')[0] + '\\' +
                                     yamls.split('\\')[-1].split('-')[0] + '-' + yamls.split('\\')[-1].split('-')[1] +
                                     '\\' + yamls.split('\\')[-1].split('-')[-1][:-5] + '\\images\\'
                                     + df.image_id[idx] +'.png')
            #im = cv2.imread('.\\Dataset\\GT_segm_new\\images\\' + df.image_id[idx] + '.png')

            x = b_box[0]
            y = b_box[1]
            w = b_box[2]
            h = b_box[3]

            # Finding midpoints
            x_centre = (x + (x + w)) / 2
            y_centre = (y + (y + h)) / 2
            img_w = 440
            img_h = 300
            # Normalization
            x_centre = x_centre / img_w
            y_centre = y_centre / img_h
            w = w / img_w
            h = h / img_h
            file_object = open('F:\\!cechy\\!images\\' + yamls.split('\\')[-1].split('-')[0] + '\\' +
                                     yamls.split('\\')[-1].split('-')[0] + '-' + yamls.split('\\')[-1].split('-')[1] +
                                     '\\' + yamls.split('\\')[-1].split('-')[-1][:-5] + '\\labels\\'
                               + df.image_id[idx] + '.txt', "a")
            bbox_n = np.array([0, x_centre, y_centre, w, h])
            # Limiting upto fix number of decimal places
            x_centre = format(x_centre, '.6f')
            y_centre = format(y_centre, '.6f')
            w = format(w, '.6f')
            h = format(h, '.6f')
            current_category = df.category_id[idx]
            file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
            file_object.close()

            #np.save('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_n)
            #np.savetxt('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_n, fmt='%i')
            #np.savetxt('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_ni)

            cropped_img = im[b_box[1]:(b_box[1]+b_box[3]), b_box[0]:(b_box[0]+b_box[2]), :]

            isExistprepr = os.path.exists('F:\\!cechy\\!images\\' + yamls.split('\\')[-1].split('-')[0] + '\\' +
            yamls.split('\\')[-1].split('-')[0] + '-' + yamls.split('\\')[-1].split('-')[1] +
            '\\' + yamls.split('\\')[-1].split('-')[-1][:-5] + '\\images_roi\\')

            if isExistprepr is False:
                os.makedirs('F:\\!cechy\\!images\\' + yamls.split('\\')[-1].split('-')[0] + '\\' +
                yamls.split('\\')[-1].split('-')[0] + '-' + yamls.split('\\')[-1].split('-')[1] +
                '\\' + yamls.split('\\')[-1].split('-')[-1][:-5] + '\\images_roi\\')

            cv2.imwrite('F:\\!cechy\\!images\\' + yamls.split('\\')[-1].split('-')[0] + '\\' +
                yamls.split('\\')[-1].split('-')[0] + '-' + yamls.split('\\')[-1].split('-')[1] +
                '\\' + yamls.split('\\')[-1].split('-')[-1][:-5] + '\\images_roi\\'
                        + df.image_id[idx] + '.png', cropped_img)

            #im_lbl = cv2.imread('.\\Dataset\\GT_segm_new\\labels\\' + df.image_id[idx] + '.png')
            #cropped_img_lbl = im_lbl[b_box[1]:(b_box[1]+b_box[3]), b_box[0]:(b_box[0]+b_box[2]), :]
            #cv2.imwrite('.\\Dataset\\GT_segm_new\\cropped\\labels\\' + df.image_id[idx] + '.png', cropped_img_lbl)


