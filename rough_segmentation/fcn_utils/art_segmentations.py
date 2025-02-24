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
import imutils

def mouth_segmentation(im):
    im_lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    th_lab, im_th_mask_ok = cv2.threshold(im_lab[:, :, 1], np.median(im_lab[:, :, 1]), 255, cv2.THRESH_BINARY)
    return im_th_mask_ok

def teeth_segmentation(im):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    pixel_values = im_hsv.reshape((-1, 3))  ### im
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(im.shape)
    th_lab, im_th_mask_teeth = cv2.threshold(cv2.normalize(src=segmented_image[:, :, 0], dst=None, alpha=0, beta=1,
                                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                                          0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return im_th_mask_teeth

def mouth_presegmentation(datax, im, i):
    """ Mouth segmentation
        datax - coordinates from yolo
        im - mouth ROI based on yolo
    """
    """ x1y1 x2y2 """
    w = int(datax[i, 2]) - int(datax[i, 0])
    h = int(datax[i, 3]) - int(datax[i, 1])

    """ mouth ROI """
    if w == 0 or h == 0:
        print('roi too small or none')
        img_zeros_b, phi_mask_01, final_mask = []
    elif w > 500 or h > 500:
        print('roi too small or none')
        img_zeros_b, phi_mask_01, final_mask = []
    else:
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

        im_mouth = im[dims1[0]:dims2[0], dims1[1]:dims2[1], :]

        im_mouth_roi = im_mouth[8:-8, round(im_mouth.shape[1] / 2) - 8:round(im_mouth.shape[1] / 2) + 8, :]
        im_mouth_roi_m = (2 * im_mouth_roi[:, :, 1] - im_mouth_roi[:, :, 2] - 0.5 * im_mouth_roi[:, :, 0]) / 4
        r = []
        g = []
        b = []
        all = []
        for ix in range(0, im_mouth_roi.shape[0]):
            r.append(np.mean(im_mouth_roi[ix, :, 0]))
            g.append(np.mean(im_mouth_roi[ix, :, 1]))
            b.append(np.mean(im_mouth_roi[ix, :, 2]))
            all.append(np.mean(im_mouth_roi_m[ix, :]))

        """ R increased in areas of mouth - G-R -> mouth area highlighted """
        a = cv2.subtract(im_mouth[:, :, 0], im_mouth[:, :, 1])

        th_a, im_th_a = cv2.threshold(a, np.median(a), 255, cv2.THRESH_BINARY)
        # fuzzy

        im_color = (2 * im_mouth[:, :, 1] - im_mouth[:, :, 2] - 0.5 * im_mouth[:, :, 0]) / 4
        img2process = np.array((im_color - np.min(im_color)) / (np.max(im_color) - np.min(im_color)))
        mean_to_fuzzy = np.mean(img2process[im_th_a > 0])
        std_to_fuzzy = np.std(img2process[im_th_a > 0])
        img2process = img2process.flatten()

        im_gauss = gaussmf(img2process, mean_to_fuzzy - std_to_fuzzy, std_to_fuzzy * (0.75))  # intensity_pixel
        im_reshape = np.reshape(im_gauss, [im_mouth[:, :, 1].shape[0], im_mouth[:, :, 1].shape[1]])
        im_reshape = cv2.normalize(im_reshape, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im_reshape_mouth = im_reshape.astype(np.uint8) # im3

        img2process = np.array((im_color - np.min(im_color)) / (np.max(im_color) - np.min(im_color)))
        mean_to_fuzzy = np.mean(img2process[im_th_a > 0])
        std_to_fuzzy = np.std(img2process[im_th_a > 0])
        img2process = img2process.flatten()
        im_gauss = gaussmf(img2process, mean_to_fuzzy - std_to_fuzzy, std_to_fuzzy)  # intensity_pixel
        im4 = np.reshape(im_gauss, [im_mouth[:, :, 1].shape[0], im_mouth[:, :, 1].shape[1]])
        im4 = cv2.normalize(im4, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask_4_teeth = im4.astype(np.uint8)

        # mask to active contour (rectangle)
        im_th_mask_ok = np.ones(a.shape)
        im_th_mask_ok[0:5, :] = 0
        im_th_mask_ok[-5:, :] = 0
        im_th_mask_ok[:, 0:5] = 0
        im_th_mask_ok[:, -5:] = 0

        """ Active Contour segmentation """
        params = gourd_params(im_reshape_mouth, im_th_mask_ok, "mouth")  # ls)#anisotropic_diffusion(abs(255-, 1)
        phi = find_lsf(**params)
        th_phi, phi_mask = cv2.threshold(cv2.normalize(src=phi, dst=None, alpha=0, beta=1,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                                         0, 2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        phi_mask = morphology.remove_small_objects(phi_mask)
        phi_mask = morphology.binary_opening(phi_mask, morphology.disk(5))
        phi_mask = phi_mask.astype('float64')
        phi_mask[phi_mask == 254] = -2
        phi_mask[phi_mask == 0] = -2

        # cut the background
        im_mouth_segm = im_mouth
        im_mouth_segm[phi_mask == -2] = 0

        """ interior mouth contour detection """

        mask_binary = phi_mask
        mask_binary[mask_binary == -2] = 0
        mask_binary[mask_binary == 2] = 255

        mask_binary = morphology.binary_opening(mask_binary, morphology.disk(5))

        # find contour
        contours, hierarchy = cv2.findContours(mask_binary.astype('uint8'), 1, 1)
        cnt = contours
        big_contour = []
        maxop = 0
        for i in cnt:
            areas = cv2.contourArea(i)
            if areas > maxop:
                maxop = areas
                big_contour = i
        try:
            mode_error = False
            ((centx, centy), (width, height), angle_rot) = cv2.fitEllipse(big_contour)
        except:
            mode_error = True

        if mode_error == False:

            im_mouth_in = im_mouth[y:y + h, x:x + w, :]
            image_changed_colors2 = im_mouth_in[int(im_mouth_in.shape[0] / 4):int(im_mouth_in.shape[0]) - int(im_mouth_in.shape[0] / 4),
                    int(im_mouth_in.shape[1] / 4):int(im_mouth_in.shape[1]) - int(im_mouth_in.shape[1] / 4)]

            img2process = im_mouth[:, :, 0].flatten()
            im2 = gaussmf(img2process, np.mean(image_changed_colors2) / 2, np.std(image_changed_colors2) / 2)  # intensity_pixel
            im3 = np.reshape(im2, [im_mouth[:, :, 1].shape[0], im_mouth[:, :, 1].shape[1]])
            im3 = cv2.normalize(im3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            im3 = im3.astype(np.uint8)

            im3[0:5, :] = 0
            im3[:, 0:5] = 0
            im3[-5:, :] = 0
            im3[:, -5:] = 0

            # initialize LSF as binary step function
            c0 = 2
            initial_lsf = c0 * np.ones(imag_64.shape)
            # generate the initial region R0 as two rectangles
            initial_lsf[Rotated_shifted > 0] = -c0

            kernel = np.ones((5, 1), np.uint8)
            im3[morphology.erosion(phi_mask, kernel) == 0] = 0
            img = np.interp(im3, [np.min(im3), np.max(im3)], [0, 255])
            img[phi_mask == 0] = 0
            params = gourd_params(img,
                                      initial_lsf, "out")
            params['iter_outer'] = 20
            params['alfa'] = 4
            params['lmda'] = 3
            params['sigma'] = 1.5
            params['epsilon'] = 1.5

            phi_mouth = find_lsf(**params)
            th_a, im_th_a = cv2.threshold(phi_mouth, 0, 1, cv2.THRESH_BINARY)
            im_th_a_org = im_th_a

            phi_mask_final = phi_mask.astype('uint8')
            final_mask = cv2.subtract(phi_mask_final.astype('int'), ~im_th_a_org.astype('int'))

            im_mouth_cropped = im_mouth[y:y + h, x:x + w]

        else:
            im_th_a_org = np.zeros((im_mouth.shape[0], im_mouth.shape[1]))
            phi_mask_final = np.zeros((im_mouth.shape[0], im_mouth.shape[1]))
            final_mask = np.zeros((im_mouth.shape[0], im_mouth.shape[1]))
            mask_4_teeth = np.zeros((im_mouth.shape[0], im_mouth.shape[1]))
            im_mouth_cropped = im_mouth
            cntr_dimensions = [0, 0, 0, 0]

    return im_th_a_org, phi_mask_final, final_mask, cntr_dimensions, im_mouth_cropped, mask_4_teeth


def teeth_presegmentation(datax, im, i):
    # x1y1 x2y2
    w = int(datax[i, 2]) - int(datax[i, 0])
    h = int(datax[i, 3]) - int(datax[i, 1])
    if w == 0 or h == 0:
        print('roi too small or none')
        im_bw = []
    elif w > 500 or h > 500:
        print('roi too small or none')
        im_bw = []
    else:
        im_teeth = im[int(datax[i, 1]):(int(datax[i, 1]) + h), int(datax[i, 0]):(int(datax[i, 0]) + w), :]
        im_teeth = cv2.normalize(src=im_teeth, dst=None, alpha=0, beta=1,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        im_teeth_roi0 = im_teeth[:, 0:30, :]
        im_teeth_roi1 = im_teeth[:, 31:round(im_teeth.shape[1] / 2), :]
        im_teeth_roi2 = im_teeth[:, round(im_teeth.shape[1] / 2) + 8:round(im_teeth.shape[1] / 2) + 8 + 30, :]
        im_teeth_roi3 = im_teeth[:, round(im_teeth.shape[1] / 2) + 8 + 30:, :]

        r0 = []
        g0 = []
        b0 = []
        for ix0 in range(0, im_teeth_roi0.shape[0]):
            r0.append(np.mean(im_teeth_roi0[ix0, :, 0]))
            g0.append(np.mean(im_teeth_roi0[ix0, :, 1]))
            b0.append(np.mean(im_teeth_roi0[ix0, :, 2]))
        r1 = []
        g1 = []
        b1 = []
        for ix1 in range(0, im_teeth_roi0.shape[0]):
            r1.append(np.mean(im_teeth_roi0[ix1, :, 0]))
            g1.append(np.mean(im_teeth_roi0[ix1, :, 1]))
            b1.append(np.mean(im_teeth_roi0[ix1, :, 2]))
        r2 = []
        g2 = []
        b2 = []
        for ix2 in range(0, im_teeth_roi2.shape[0]):
            r2.append(np.mean(im_teeth_roi2[ix2, :, 0]))
            g2.append(np.mean(im_teeth_roi2[ix2, :, 1]))
            b2.append(np.mean(im_teeth_roi2[ix2, :, 2]))
        r3 = []
        g3 = []
        b3 = []
        for ix3 in range(0, im_teeth_roi3.shape[0]):
            r3.append(np.mean(im_teeth_roi3[ix3, :, 0]))
            g3.append(np.mean(im_teeth_roi3[ix3, :, 1]))
            b3.append(np.mean(im_teeth_roi3[ix3, :, 2]))

        new_img = np.zeros((im_teeth.shape[0], im_teeth.shape[1]))
        for ix_h in range(0, im_teeth.shape[0]):
            for ix_w in range(0, im_teeth.shape[1]):
                if ((im_teeth[ix_h, ix_w, 1] - im_teeth[ix_h, ix_w, 0]) < 0):
                    new_img[ix_h, ix_w] = 0
                else:
                    new_img[ix_h, ix_w] = im_teeth[ix_h, ix_w, 1] - im_teeth[ix_h, ix_w, 0]
                    if new_img[ix_h, ix_w] < 0:
                        print(ix_w + ' + ' + ix_h)

        mask_img = new_img > 0.01
        mask_img = mask_img

        # mask_img = morphology.binary_dilation(mask_img, morphology.disk(2))
        mask_img = mask_img.astype('uint8') * 2
        mask_img[mask_img == 0] = -2
        mask_img[0:2, :] = -2
        mask_img[-2:, :] = -2
        mask_img[:, 0:2] = -2
        mask_img[:, -2:] = -2
        mask_img = cv2.normalize(mask_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        mask_img[mask_img == 0] = 2
        mask_img[mask_img == 255] = -2
        params = gourd_params(
            cv2.normalize(new_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
            mask_img, "teeth")
        params['iter_outer'] = 10  # ls)#anisotropic_diffusion(abs(255-, 1)
        try:
            phi = find_lsf(**params)
            thresh, im_bw = cv2.threshold(phi, 0, 255, cv2.THRESH_BINARY)
        except:
            im_bw = np.zeros(new_img.shape)
    return im_bw


def teeth_presegmentation_ROI(datax, im, i, ix, mask, mask_AC, xywh):
    # x1y1 x2y2
    w = int(datax[ix, 2]) - int(datax[ix, 0])
    h = int(datax[ix, 3]) - int(datax[ix, 1])
    if w == 0 or h == 0:
        print('error')
        im_bw = []
    elif w > 500 or h > 500:
        print('error')
        im_bw = []
    else:

        dims1 = []
        dims2 = []
        dims1.append(int(datax[ix, 1]) - 3)
        dims1.append(int(datax[ix, 0]) - 3)
        dims2.append(int(datax[ix, 1]) + h + 6)
        dims2.append(int(datax[ix, 0]) + w + 6)

        if dims1[0] < 0:
            dims1[0] = 0
        if dims1[1] < 0:
            dims1[1] = 0

        if dims2[0] > im.shape[0]:
            dims2[0] = im.shape[0]
        if dims2[1] > im.shape[1]:
            dims2[1] = im.shape[1]

        c0 = 2
        initial_lsf = c0 * np.ones((im.shape[0], im.shape[1]))
        initial_lsf[int(datax[i, 1]):(int(datax[i, 1]) + int(datax[i, 3]) - int(datax[i, 1])),
        int(datax[i, 0]):(int(datax[i, 0]) + int(datax[i, 2]) - int(datax[i, 0]))] = -c0

        initial_lsf = initial_lsf[int(datax[ix, 1]) - 3:(int(datax[ix, 1]) + h + 6),
                      int(datax[ix, 0] - 3):(int(datax[ix, 0]) + w + 6)]
        maska = 255 - mask
        maska[mask_AC == 0] = 0
        img2 = morphology.opening(maska, morphology.disk(2))
        img2[initial_lsf == 2] = 0
        img = np.interp(img2, [np.min(img2), np.max(img2)], [0, 255])

        imgTF = img > np.sum(img) / np.count_nonzero(img) - np.std(img) / 2
        img[imgTF == False] = 0
        params = gourd_params(img,
                                  initial_lsf, "out")
        params['iter_outer'] = 25
        params['alfa'] = 6
        params['lmda'] = 4
        params['sigma'] = 1.5
        params['epsilon'] = 2

        phiii = find_lsf(**params)

        th_a, im_th_a = cv2.threshold(phiii, 0, 255, cv2.THRESH_BINARY)
        im_th_a = morphology.binary_opening(im_th_a, morphology.disk(2))
        im_th_a = 255 - im_th_a

    return im_th_a


def tongue_presegmentation(datax, im, i, ix):
    # x1y1 x2y2
    w = int(datax[ix, 2]) - int(datax[ix, 0])
    h = int(datax[ix, 3]) - int(datax[ix, 1])
    if w == 0 or h == 0:
        print('error')
        im_bw = []
    elif w > 500 or h > 500:
        print('error')
        im_bw = []
    else:
        dims1 = []
        dims2 = []
        dims1.append(int(datax[ix, 1]) - 3)
        dims1.append(int(datax[ix, 0]) - 3)
        dims2.append(int(datax[ix, 1]) + h + 6)
        dims2.append(int(datax[ix, 0]) + w + 6)

        if dims1[0] < 0:
            dims1[0] = 0
        if dims1[1] < 0:
            dims1[1] = 0

        if dims2[0] > im.shape[0]:
            dims2[0] = im.shape[0]
        if dims2[1] > im.shape[1]:
            dims2[1] = im.shape[1]

        im_tongue = im[dims1[0]:dims2[0], dims1[1]:dims2[1]]
        im_tongue = cv2.normalize(src=im_tongue, dst=None, alpha=0, beta=1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        initial_lsf = c0 * np.ones((im.shape[0], im.shape[1]))
        initial_lsf[int(datax[i, 1]):(int(datax[i, 1]) + int(datax[i, 3]) - int(datax[i, 1])),
        int(datax[i, 0]):(int(datax[i, 0]) + int(datax[i, 2]) - int(datax[i, 0]))] = -c0

        initial_lsf = initial_lsf[int(datax[ix, 1]) - 3:(int(datax[ix, 1]) + h + 6),
                      int(datax[ix, 0]) - 3:(int(datax[ix, 0]) + w + 6)]
        params = gourd_params(imim[:, :, 0],
                                  initial_lsf, "out")
        params['iter_outer'] = 25
        params['alfa'] = 5
        params['lmda'] = 3
        params['sigma'] = 1.5
        try:
            phi_tongue = find_lsf(**params)
        except:
            phi_tongue = np.zeros(imim[:, :, 0].shape)
        th_a, im_th_a = cv2.threshold(phi_tongue, 0, 255, cv2.THRESH_BINARY)
    return im_th_a