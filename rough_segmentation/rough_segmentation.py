from utils.general import xywhn2xyxy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

from lv_set.find_lsf import find_lsf
from lv_set.potential_func import *
from lv_set.show_fig import draw_all
from scipy import ndimage
from skimage import morphology, filters, segmentation

from skfuzzy.membership import gauss2mf, gaussmf, dsigmf
from skimage import morphology
from fcn_utils.art_segmentations import mouth_segmentation, teeth_segmentation, mouth_presegmentation, teeth_presegmentation, teeth_presegmentation_ROI, tongue_presegmentation


def gourd_params(img, img_mask, mode):
    """ initialize LSF """
    c0 = 2
    if mode is "out":
        initial_lsf = img_mask
        timestep = 1  # time step
        iter_inner = 5
        iter_outer = 20
        lmda = 5  # coefficient of the weighted length term L(phi)
        alfa = -3  # coefficient of the weighted area term A(phi) -3
        epsilon = 1.5  # parameter that specifies the width of the DiracDelta function
        sigma = 0.8  # scale parameter in Gaussian kernel
    elif mode is "teeth":
        initial_lsf = img_mask
        timestep = 1  # time step
        iter_inner = 1
        iter_outer = 25
        lmda = 5  # coefficient of the weighted length term L(phi)
        alfa = -5  # coefficient of the weighted area term A(phi) -3
        epsilon = 1.5  # parameter that specifies the width of the DiracDelta function
        sigma = 0.8  # scale parameter in Gaussian kernel
    elif mode is "mouth":
        img_mask = img_mask.astype('uint8')
        img_mask = ndimage.binary_fill_holes(img_mask)
        img_mask = morphology.remove_small_objects(img_mask, 5)
        initial_lsf = img_mask
        initial_lsf = c0 * initial_lsf
        initial_lsf = initial_lsf.astype('float64')
        timestep = 1  # time step
        iter_inner = 5
        iter_outer = 25
        lmda = 5  # coefficient of the weighted length term L(phi)
        alfa = -5  # coefficient of the weighted area term A(phi) -3
        epsilon = 2  # parameter that specifies the width of the DiracDelta function
        sigma = 0.5  # scale parameter in Gaussian kernel

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': timestep,  # time step
        'iter_inner': iter_inner,
        'iter_outer': iter_outer,
        'lmda': lmda,  # coefficient of the weighted length term L(phi)
        'alfa': alfa,  # coefficient of the weighted area term A(phi) -3
        'epsilon': epsilon,  # parameter that specifies the width of the DiracDelta function
        'sigma': sigma,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }




""" CHOOSE DIR PATH """
images_in_folder = glob.glob(
    '.\\exp_detect\\labels\\' + '*.txt')
mode_full_face = 1
counter_to_start = -1

for files in images_in_folder:
    filenames = files.split('\\')[-1]
    txt_load = np.loadtxt(
        files)
    datax = xywhn2xyxy(txt_load[:, 1:], w=880, h=600)
    im = cv2.imread('.\\test_fin\\' + filenames.replace('txt','png'))

    img2show = cv2.imread(
        '.\\test\\' + filenames.replace('txt', 'png'))
    masks_mouth = []
    masks_mouth_idx = []
    masks_teeth = []
    masks_teeth_idx = []

    for i in range(0, datax.shape[0]):
        counter_to_start = counter_to_start + 1
        print('mosaic '+str(i))
        if counter_to_start >= 0:
            label = txt_load[i, 0]

            """ 0 in yolo == MOUTH SEGMENTATION """
            if label == 0:
                labels_to_visit = []
                mask_inter, mask_outer,mask_both, cntr_dimensions, image_rotated, mask_4_teeth = mouth_presegmentation(
                    datax, im, i)
                masks_mouth.append(mask_both)
                masks_mouth_idx.append(datax[i, :])

                w = int(datax[i, 2]) - int(datax[i, 0])
                h = int(datax[i, 3]) - int(datax[i, 1])

                for ix in range(0, datax.shape[0]):
                    if ix == i:
                        print('same')
                    else:
                        if int(datax[i, 1]) <= int(datax[ix, 1]) <= (int(datax[i, 1]) + h) and int(datax[i, 0]) <= int(
                                datax[ix, 0]) <= (int(datax[i, 0]) + w):
                            labels_to_visit.append(ix)

                masks_teeths = []
                coords_teeths = []
                masks_tongues = []
                coords_tongues = []
                flag_tongue = datax.shape[0] + 1
                flag_teeth = datax.shape[0] + 1
                for ix2 in range(0, len(labels_to_visit)):

                    """ 1 in yolo == TEETH SEGMENTATION """
                    if txt_load[labels_to_visit[ix2], 0] == 1:
                        mask_teeth = teeth_presegmentation_ROI(datax, im, labels_to_visit[ix2], i, mask_4_teeth,
                                                                   mask_outer, cntr_dimensions)
                        mask_teeth[mask_teeth == 0] = 1
                        mask_teeth[mask_teeth == 255] = 0

                        masks_teeths.append(mask_teeth)
                        coords_teeths.append(
                                [int(datax[i, 1]) - 3, int(datax[i, 1]) + int(datax[i, 3]) - int(datax[i, 1]) + 6,
                                 int(datax[i, 0] - 3), (int(datax[i, 0]) + int(datax[i, 2]) - int(datax[i, 0]) + 6)])

                        flag_teeth = i
                        counter_teeth = i

                    """ 2 in yolo == TONGUE SEGMENTATION """
                    if txt_load[labels_to_visit[ix2], 0] == 2:
                        im = cv2.imread(
                            '.\\test_fin\\' + filenames.replace(
                                'txt',
                                'png'))
                        mask_tongue = tongue_presegmentation(np.round(datax), im, labels_to_visit[ix2], i)
                        mask_tongue[mask_tongue == 0] = 1
                        mask_tongue[mask_tongue == 255] = 0

                        coords_tongues.append(
                            [int(datax[i, 1]) - 3, int(datax[i, 1]) + int(datax[i, 3]) - int(datax[i, 1]) + 6,
                             int(datax[i, 0] - 3), (int(datax[i, 0]) + int(datax[i, 2]) - int(datax[i, 0]) + 6)])
                        masks_tongues.append(mask_tongue)
                        flag_tongue = i

                    dims1 = []
                    dims2 = []
                    dims1.append(int(datax[i, 1]) - 3)
                    dims1.append(int(datax[i, 0]) - 3)
                    dims2.append(int(datax[i, 1]) + h + 6)
                    dims2.append(int(datax[i, 0]) + w + 6)

                    if dims1[0] < 0:
                        dims1[0] = 0
                    if dims1[1] < 0:
                        dims1[1] = 0

                    if dims2[0] > im.shape[0]:
                        dims2[0] = im.shape[0]
                    if dims2[1] > im.shape[1]:
                        dims2[1] = im.shape[1]

                    image_labels = np.zeros((img2show[dims1[0]:dims2[0], dims1[1]:dims2[1]].shape[0],
                                             img2show[dims1[0]:dims2[0], dims1[1]:dims2[1]].shape[1]))
                    image_labels[mask_both == 3] = 1

                    if flag_teeth == i:
                        for m_t in range(0, len(masks_teeths)):
                            m_teeth_2_f = np.zeros(image_labels.shape)
                            shape_size = m_teeth_2_f[dims1[0] - coords_teeths[m_t][0]:coords_teeths[m_t][1],
                                            dims1[1] - coords_teeths[m_t][2]:coords_teeths[m_t][3]].shape
                            m_teeth_2_f[dims1[0] - coords_teeths[m_t][0]:coords_teeths[m_t][1],
                            dims1[1] - coords_teeths[m_t][2]:coords_teeths[m_t][3]] = cv2.resize(masks_teeths[m_t],
                                                                                                     (shape_size[1],
                                                                                                      shape_size[0]))
                            image_labels[m_teeth_2_f == 0] = 2

                    if flag_tongue == i:
                        for m_ton in range(0, len(masks_tongues)):
                                # plt.contour(masks_tongues[m_ton], [0], colors='m', linewidth=2)
                            m_tongue_2_f = np.zeros(image_labels.shape)
                            shape_size = m_tongue_2_f[dims1[0] - coords_tongues[m_ton][0]:coords_tongues[m_ton][1],
                                             dims1[1] - coords_tongues[m_ton][2]:coords_tongues[m_ton][3]
                                                          ].shape
                            m_tongue_2_f[dims1[0] - coords_tongues[m_ton][0]:coords_tongues[m_ton][1],
                            dims1[1] - coords_tongues[m_ton][2]:coords_tongues[m_ton][3]] = cv2.resize(masks_tongues[m_ton],
                                                                                                           (shape_size[1], shape_size[0]))
                            image_labels[m_tongue_2_f == 1] = 3

                    cv2.imwrite(
                        '.\\3_labels_test\\labels\\' + filenames[
                                                                                                                        :-4] + '-' + str(
                            i) + '.png', image_labels)
                    cv2.imwrite(
                        '.\\3_labels_test\\images\\' + filenames[
                                                                                                                        :-4] + '-' + str(
                            i) + '.png', img2show[dims1[0]:dims2[0], dims1[1]:dims2[1], :])
            """ TEETH SEGMENTATION """
            if label == 5:
                mask_teeth = teeth_presegmentation(datax, im, i)
                masks_teeth.append(mask_teeth)
                masks_teeth_idx.append(datax[i, :])

    print(' -- image ' + filenames + ' done --')

