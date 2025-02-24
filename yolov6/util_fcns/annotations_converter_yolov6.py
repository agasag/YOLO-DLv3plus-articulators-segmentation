'''
# Crop images to mouth ROIs according to YOLOv6 results
'''

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import shutil

'''def class_to_color(class_id):
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
        draw_bounding_box(img, row)'''


error = []
# point YAML files covering images
yaml_files = glob.glob('D:\\!cechy\\!data_05022024\\yolo_yaml_files_part7\\*.yaml')
for yamls in yaml_files:
    # var for storage the biggest area (will be used to crop unYOLOed images)
    area = None
    try:
        tempSCORE = 0
        kid_yaml = yamls.split('\\')[-1]
        kid_yaml_info = kid_yaml.split('-')
        kidID = kid_yaml_info[0]
        kidRecID = kid_yaml_info[1]
        kidWord = kid_yaml_info[2]
        kidCamera = kid_yaml_info[-1]

        # path to images
        """if (int(kidID) >=30 and int(kidID) <= 199):
            pathImgDir = 'D:\\!cechy\\!data_05022024\\yolo_preprocessed_frames_part2\\' + kidID + '\\' + kidID + '-' + kidRecID + \
                         '\\' + kidWord + '\\' + kidCamera[:-5]
        elif int(kidID) >= 229 and int(kidID) <= 237:
            pathImgDir = 'D:\\!cechy\\!data_05022024\\yolo_preprocessed_frames_part2\\' + kidID + '\\' + kidID + '-' + kidRecID + \
                         '\\' + kidWord + '\\' + kidCamera[:-5]
        elif int(kidID) >= 200 and int(kidID) <= 228:
            pathImgDir = 'D:\\!cechy\\!data_05022024\\yolo_preprocessed_frames\\' + kidID + '\\' + kidID + '-' + kidRecID + \
                         '\\' + kidWord + '\\' + kidCamera[:-5]"""
        pathImgDir = 'D:\\!cechy\\!data_05022024\\yolo_preprocessed_frames\\' + kidID + '\\' + kidID + '-' + kidRecID + \
                         '\\' + kidWord + '\\' + kidCamera[:-5]
        print('Vid ' + yamls.split('\\')[-1][:-5] + ' processing...')

        # read YOLO json file with the results
        df = pd.read_json(
            'F:\\runs\\val\\_21_04_2024\\' + yamls.split('\\')[-1][:-5] + '\\predictions.json')
        allImagesIDs = df['image_id']

        isExistprepr = os.path.exists(pathImgDir + '\\images_roi\\')
        if isExistprepr is True:
            shutil.rmtree(pathImgDir + '\\images_roi\\')

        # for each mouth YOLO ROIs (category_id = 0)
        for idx in range(0, len(df)):
            print('- ' + yamls.split('\\')[-1][:-5] + ': ' + df.image_id[idx])
            img_id = df.image_id[idx]

            if df.category_id[idx] == 0:
                flag_noBB = False # False: there is YOLO ROI; True: there's no ROI
                counterFramesIDs = 0
                frameIDs = np.where(allImagesIDs == img_id)
                # if more frames than 1
                if len(frameIDs[0]) > 1:
                    for idxFrame in frameIDs[0]:
                        if df.category_id[idxFrame] == 0:
                            if counterFramesIDs == 0:
                                # remove weak ROIs
                                if df.score[idxFrame] < 0.75:
                                    # assign empty category to omit the ROI
                                    df.loc[idxFrame, 'category_id'] = 5
                                    flag_noBB = True
                                    df.to_json(
                                        'F:\\runs\\val\\_21_04_2024\\' + yamls.split('\\')[-1][
                                                                                                 :-5] + '\\predictions.json',
                                        orient='records')
                                else:
                                    tempCAT = df.category_id[idxFrame]
                                    tempSCORE = df.score[idxFrame]
                                    tempIDX = idxFrame
                                    flag_noBB = False

                            else:
                                if tempSCORE > df.score[idxFrame]:
                                    df.loc[idxFrame, 'category_id'] = 5
                                    flag_noBB = True
                                    df.to_json(
                                        'F:\\runs\\val\\_21_04_2024\\' + yamls.split('\\')[-1][
                                                                                                 :-5] + '\\predictions.json',
                                        orient='records')
                                else:
                                    tempSCORE = df.score[idxFrame]
                                    tempIDX = idxFrame
                                    flag_noBB = False
                            counterFramesIDs = counterFramesIDs+1
                else:
                    idxFrame = frameIDs[0][0]
                    # remove weak ROIs
                    if df.score[idxFrame] < 0.75:
                        df.loc[idxFrame, 'category_id'] = 5
                        df.to_json('F:\\runs\\val\\_21_04_2024\\' + yamls.split('\\')[-1][
                                                                                            :-5] + '\\predictions.json',
                                        orient='records')
                        flag_noBB = True
                    else:
                        tempCAT = df.category_id[idxFrame]
                        tempSCORE = df.score[idxFrame]
                        tempIDX = idxFrame
                        flag_noBB = False

                # calculate and storage areas
                if flag_noBB is False:
                    b_box = np.round(df.bbox[idx]).astype(int)

                    if area is not None:
                        area = b_box[2]*b_box[3]
                        if b_box[2]*b_box[3] > temp_area_df[2]*temp_area_df[3]:
                            temp_area_df = b_box
                    else:
                        temp_area_df = b_box

                    # read images
                    im = cv2.imread(pathImgDir + '\\images\\' + df.image_id[idx] + '.png')

                    x = b_box[0]
                    y = b_box[1]
                    w = b_box[2]
                    h = b_box[3]

                    # if BB is valid (none of elements is zero)
                    if x != 0 and y != 0 and w != 0 and h != 0:
                        df_temp = df.bbox[idx]
                        x_centre = (x + (x + w)) / 2
                        y_centre = (y + (y + h)) / 2
                        img_w = 440
                        img_h = 300

                        ''' is it necessary? '''
                        # Normalization
                        x_centre = x_centre / img_w
                        y_centre = y_centre / img_h
                        w = w / img_w
                        h = h / img_h
                        file_object = open(pathImgDir + '\\labels\\' + df.image_id[idx] + '.txt', "a")
                        bbox_n = np.array([0, x_centre, y_centre, w, h])

                        # Limiting upto fix number of decimal places
                        x_centre = format(x_centre, '.6f')
                        y_centre = format(y_centre, '.6f')
                        w = format(w, '.6f')
                        h = format(h, '.6f')
                        current_category = df.category_id[idx]
                        file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
                        file_object.close()
                        ''' up yo here '''

                        # Crop the ROI
                        cropped_img = im[b_box[1]:(b_box[1] + b_box[3]), b_box[0]:(b_box[0] + b_box[2]), :]
                        #cropped_img = im[b_box[1] - 5:(b_box[1] - 5 + b_box[3] + 5),
                        #             b_box[0] - 5:(b_box[0] - 5 + b_box[2] ),:]
                        isExistprepr = os.path.exists(pathImgDir + '\\images_roi\\')

                        if isExistprepr is False:
                            os.makedirs(pathImgDir + '\\images_roi\\')
                        if os.path.isfile(pathImgDir + '\\images_roi\\' + df.image_id[idx] + '.png') is True:
                            os.remove(pathImgDir + '\\images_roi\\' + df.image_id[idx] + '.png')
                        cv2.imwrite(pathImgDir + '\\images_roi\\' + df.image_id[idx] + '.png', cropped_img)

                    # if there is some zero elements - take the biggest area ROI
                    else:

                        b_box = np.round(df_temp).astype(int)

                        im = cv2.imread(pathImgDir + '\\images\\' + df.image_id[idx] + '.png')

                        x = b_box[0]
                        y = b_box[1]
                        w = b_box[2]
                        h = b_box[3]

                        if x != 0 and y != 0 and w != 0 and h != 0:

                            x_centre = (x + (x + w)) / 2
                            y_centre = (y + (y + h)) / 2
                            img_w = 440
                            img_h = 300

                            ''' is it necessary? again '''
                            # Normalization
                            x_centre = x_centre / img_w
                            y_centre = y_centre / img_h
                            w = w / img_w
                            h = h / img_h
                            file_object = open(pathImgDir + '\\labels\\' + df.image_id[idx] + '.txt', "a")
                            bbox_n = np.array([0, x_centre, y_centre, w, h])
                            # Limiting upto fix number of decimal places
                            x_centre = format(x_centre, '.6f')
                            y_centre = format(y_centre, '.6f')
                            w = format(w, '.6f')
                            h = format(h, '.6f')
                            current_category = df.category_id[idx]
                            file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
                            file_object.close()
                            '''  '''

                            # crop image to mouth ROI
                            cropped_img = im[b_box[1]:(b_box[1] + b_box[3]), b_box[0]:(b_box[0] + b_box[2]), :]
                            #cropped_img = im[b_box[1] :(b_box[1] + b_box[3] ),
                            #              b_box[0] :(b_box[0]  + b_box[2] + 15), :]

                            isExistprepr = os.path.exists(pathImgDir + '\\images_roi\\')

                            if isExistprepr is False:
                                os.makedirs(pathImgDir + '\\images_roi\\')
                            if os.path.isfile(pathImgDir + '\\images_roi\\' + df.image_id[idx] + '.png') is True:
                                os.remove(pathImgDir + '\\images_roi\\' + df.image_id[idx] + '.png')
                            cv2.imwrite(pathImgDir + '\\images_roi\\' + df.image_id[idx] + '.png', cropped_img)
                        print('ha!')
                        count = 1

            # path joining version for other paths
        DIR1 = pathImgDir + '\\images\\'
        DIR2 = pathImgDir + '\\images_roi\\'
        DIR1len = len([name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))])
        DIR2len = len([name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))])
        DIR1gen = [name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
        DIR2gen = [name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]

        df_temp2 = df

        if DIR1len > DIR2len:
            for im_name in DIR1gen:
                if im_name not in DIR2gen:
                    b_box = np.round(temp_area_df).astype(int) #np.round(df_temp).astype(int)

                    im = cv2.imread(pathImgDir + '\\images\\' + im_name)

                    x = b_box[0] - 15
                    y = b_box[1] - 15
                    w = b_box[2]+30
                    h = b_box[3]+30

                    if x != 0 and y != 0 and w != 0 and h != 0:

                        x_centre = (x + (x + w)) / 2
                        y_centre = (y + (y + h)) / 2
                        img_w = 440
                        img_h = 300
                        # Normalization
                        x_centre = x_centre / img_w
                        y_centre = y_centre / img_h
                        w = w / img_w
                        h = h / img_h
                        """file_object = open(pathImgDir + '\\labels\\' + im_name[:-4] + '.txt', "a")
                        bbox_n = np.array([0, x_centre, y_centre, w, h])
                        # Limiting upto fix number of decimal places
                        x_centre = format(x_centre, '.6f')
                        y_centre = format(y_centre, '.6f')
                        w = format(w, '.6f')
                        h = format(h, '.6f')
                        current_category = df.category_id[idx]
                        file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
                        file_object.close()"""

                        # np.save('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_n)
                        # np.savetxt('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_n, fmt='%i')
                        # np.savetxt('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_ni)

                        #cropped_img = im[b_box[1] - 10:(b_box[1] - 10 + b_box[3] + 30),
                        #              b_box[0] - 10:(b_box[0] - 10 + b_box[2] + 30),:]
                        cropped_img = im[b_box[1]:(b_box[1] + b_box[3]), b_box[0]:(b_box[0] + b_box[2]), :]
                        #cropped_img = im[b_box[1] :(b_box[1] + b_box[3] ),
                        #              b_box[0]:(b_box[0] + b_box[2] + 15),:]

                        isExistprepr = os.path.exists(pathImgDir + '\\images_roi\\')

                        if isExistprepr is False:
                            os.makedirs(pathImgDir + '\\images_roi\\')
                        if os.path.isfile(pathImgDir + '\\images_roi\\' + im_name[:-4] + '.png') is True:
                            os.remove(pathImgDir + '\\images_roi\\' + im_name[:-4] + '.png')
                        cv2.imwrite(pathImgDir + '\\images_roi\\' + im_name[:-4] + '.png', cropped_img)

                        df.loc[len(df)] = {'image_id': im_name[:-4], 'category_id': 0, 'bbox': df_temp, 'score': 0.8}
                        df.to_json('F:\\runs\\val\\_21_04_2024\\' + yamls.split('\\')[-1][
                                                                                            :-5] + '\\predictions.json',
                                        orient='records')
        #
        #df.to_json('D:\\!cechy\\YOLOv6\\tools\\runs\\val\\new4\\' + yamls.split('\\')[-1][:-5] + '\\predictions.json')
    except:
        error.append(pathImgDir)

        """if df.score[idx] > temp_img.score:
            pass
        
        if idx > 0:
            if img_id == temp_img.image_id:
                if df.score[idx] > temp_img.score:
                    if df.category_id[idx] == 0:
                        b_box = np.round(df.bbox[idx]).astype(int)
                        im = cv2.imread(pathImgDir + '\\images\\' + df.image_id[idx] + '.png')

                        x = b_box[0]
                        y = b_box[1]
                        w = b_box[2]
                        h = b_box[3]

                        if x != 0 and y != 0 and w != 0 and h != 0:

                            x_centre = (x + (x + w)) / 2
                            y_centre = (y + (y + h)) / 2
                            img_w = 440
                            img_h = 300
                            # Normalization
                            x_centre = x_centre / img_w
                            y_centre = y_centre / img_h
                            w = w / img_w
                            h = h / img_h
                            file_object = open(pathImgDir + '\\labels\\' + df.image_id[idx] + '.txt', "a")
                            bbox_n = np.array([0, x_centre, y_centre, w, h])
                            # Limiting upto fix number of decimal places
                            x_centre = format(x_centre, '.6f')
                            y_centre = format(y_centre, '.6f')
                            w = format(w, '.6f')
                            h = format(h, '.6f')
                            current_category = df.category_id[idx]
                            file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
                            file_object.close()

                            # np.save('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_n)
                            # np.savetxt('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_n, fmt='%i')
                            # np.savetxt('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_ni)

                            cropped_img = im[b_box[1]:(b_box[1] + b_box[3]), b_box[0]:(b_box[0] + b_box[2]), :]

                            isExistprepr = os.path.exists(pathImgDir + '\\images_roi\\')

                            if isExistprepr is False:
                                os.makedirs(pathImgDir + '\\images_roi\\')

                            cv2.imwrite(pathImgDir + '\\images_roi\\' + df.image_id[idx] + '.png', cropped_img)

                            # im_lbl = cv2.imread('.\\Dataset\\GT_segm_new\\labels\\' + df.image_id[idx] + '.png')
                            # cropped_img_lbl = im_lbl[b_box[1]:(b_box[1]+b_box[3]), b_box[0]:(b_box[0]+b_box[2]), :]
                            # cv2.imwrite('.\\Dataset\\GT_segm_new\\cropped\\labels\\' + df.image_id[idx] + '.png', cropped_img_lbl)
                        else:
                            print('ha!')
                            count = 1
        else:
            if df.category_id[idx] == 0:
                b_box = np.round(df.bbox[idx]).astype(int)
                im = cv2.imread(pathImgDir + '\\images\\' + df.image_id[idx] + '.png')

                x = b_box[0]
                y = b_box[1]
                w = b_box[2]
                h = b_box[3]

                if x != 0 and y != 0 and w != 0 and h != 0:

                    x_centre = (x + (x + w)) / 2
                    y_centre = (y + (y + h)) / 2
                    img_w = 440
                    img_h = 300
                    # Normalization
                    x_centre = x_centre / img_w
                    y_centre = y_centre / img_h
                    w = w / img_w
                    h = h / img_h
                    file_object = open(pathImgDir + '\\labels\\' + df.image_id[idx] + '.txt', "a")
                    bbox_n = np.array([0, x_centre, y_centre, w, h])
                    # Limiting upto fix number of decimal places
                    x_centre = format(x_centre, '.6f')
                    y_centre = format(y_centre, '.6f')
                    w = format(w, '.6f')
                    h = format(h, '.6f')
                    current_category = df.category_id[idx]
                    file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
                    file_object.close()

                    # np.save('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_n)
                    # np.savetxt('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_n, fmt='%i')
                    # np.savetxt('D:\\AgataS\\!phd\\_test_YOLO\\0035\\right\\labels\\' + df.image_id[idx] + '.txt', bbox_ni)

                    cropped_img = im[b_box[1]:(b_box[1] + b_box[3]), b_box[0]:(b_box[0] + b_box[2]), :]

                    isExistprepr = os.path.exists(pathImgDir + '\\images_roi\\')

                    if isExistprepr is False:
                        os.makedirs(pathImgDir + '\\images_roi\\')

                    cv2.imwrite(pathImgDir + '\\images_roi\\' + df.image_id[idx] + '.png', cropped_img)

                    # im_lbl = cv2.imread('.\\Dataset\\GT_segm_new\\labels\\' + df.image_id[idx] + '.png')
                    # cropped_img_lbl = im_lbl[b_box[1]:(b_box[1]+b_box[3]), b_box[0]:(b_box[0]+b_box[2]), :]
                    # cv2.imwrite('.\\Dataset\\GT_segm_new\\cropped\\labels\\' + df.image_id[idx] + '.png', cropped_img_lbl)
                else:
                    print('ha!')
                    count = 1
        temp_img = df.iloc[idx]"""
