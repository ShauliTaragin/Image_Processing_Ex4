import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import *
import cv2


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # initialize the disparity map which we will return
    disparity_map = np.zeros((img_r.shape[0], img_r.shape[1], disp_range[1]))

    # pass the images through a multidimensional filter toward reaching the normalized images
    before_norm_l = filters.uniform_filter(img_l, k_size)
    before_norm_r = filters.uniform_filter(img_r, k_size)

    # hold variables for iteration length
    start = disp_range[0]
    end = disp_range[1]

    # work according to the given formula to find disparity
    for shift_size in range(end - start):
        # normalized image
        norm_left_img = img_l - before_norm_l
        norm_right_img = img_r - before_norm_r
        # shift right according to formula
        right_img_after_shft = np.roll(norm_right_img, shift_size)
        disparity_map[:, :, shift_size] = filters.uniform_filter(norm_left_img * right_img_after_shft, k_size)
        # square the disparity map in the proper location before summing(i.e iterating).
        disparity_map[:, :, shift_size] **= 2

    # need to return the minimum ssd amount. That is given in the maximum disparity map which we found
    # (in the 3rd dimension which we changed)
    min_ssd_img = np.argmax(disparity_map, axis=2)
    return min_ssd_img


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # initialize the disparity map which we will return
    disparity_map = np.zeros((img_r.shape[0], img_r.shape[1], disp_range[1]))

    # pass the images through a multidimensional filter toward reaching the normalized images
    before_norm_l = filters.uniform_filter(img_l, k_size)
    before_norm_r = filters.uniform_filter(img_r, k_size)

    # hold variables for iteration length
    start = 0
    end = disp_range[1]

    # work according to the given formula to find disparity
    for shift_size in range(end - start):
        # normalized image
        norm_left_img = img_l - before_norm_l
        norm_right_img = img_r - before_norm_r

        # R_LL takes k_size and only left image so we can calculate it here before shifting right
        R_LL = filters.uniform_filter(norm_left_img ** 2, k_size)

        # shift right according to formula
        right_img_after_shft = np.roll(norm_right_img, shift_size)
        # calculate R_(LR) and R_(RR)
        R_LR = filters.uniform_filter(norm_left_img * right_img_after_shft, k_size)
        R_RR = filters.uniform_filter(right_img_after_shft ** 2, k_size)
        # calculate normalized result(according to formula) and save it in disparity_map
        disparity_map[:, :, shift_size] = R_LR / np.sqrt(R_RR * R_LL)

    # need to return the maximum RC score. That is given in the maximum disparity map which we found
    # (in the 3rd dimension which we changed)
    min_ssd_img = np.argmax(disparity_map, axis=2)
    return min_ssd_img


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    # create matrix to hold all points
    A = np.zeros((2 * len(src_pnt), 9))
    for i in range(len(src_pnt)):
        x_src, y_src = src_pnt[i]
        x_dst, y_dst = dst_pnt[i]
        A[i * 2:i * 2 + 2] = np.array([[-x_src, -y_src, -1, 0, 0, 0, x_src * x_dst, y_src * x_dst, x_dst],
                                       [0, 0, 0, -x_src, -y_src, -1, x_src * y_dst, y_src * y_dst, y_dst]])
    # svd on matrix of points which we created.
    # naming variables according to documentation
    u, sigma, vh_T = np.linalg.svd(A, full_matrices=True)
    # return to vh
    vh = np.transpose(vh_T)

    # calculate the Homography matrix
    scnd_elemnt_of_Vh = vh[:, -1]
    Homography_matrix = scnd_elemnt_of_Vh.reshape(3, 3)
    Homography_matrix /= scnd_elemnt_of_Vh[-1]

    # find the error
    error = 0
    for i in range(len(dst_pnt)):
        x, y = src_pnt[i]
        Homogeneous = np.array([x, y, 1])  # make the point homogeneous
        Homogeneous = Homography_matrix.dot(Homogeneous)
        Homogeneous /= Homogeneous[2]

        # error according to formula
        error += np.sqrt(sum(Homogeneous[0:-1] - dst_pnt[i]) ** 2)

    return Homography_matrix, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    # copy same as for image 1 and apply it to image 2
    src_p = []
    fig2 = plt.figure()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display second image
    cid2 = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    ##### Your Code Here ######
    max_x = int(max([x[0] for x in src_p]))
    max_y = int(max([x[1] for x in src_p]))
    min_x = int(min([x[0] for x in src_p]))
    min_y = int(min([x[1] for x in src_p]))

    Homography, blank = computeHomography(src_p, dst_p)
    for i in range(min_y , max_y):
        for j in range(min_x , max_x):
            Ah = np.array([j, i, 1])
            Ah = Homography.dot(Ah)
            x = Ah[0]
            y = Ah[1]
            Ah /= Ah[2]
            dst_img[int(Ah[1]), int(Ah[0])] = src_img[i, j]
    plt.imshow(dst_img)
    plt.show()

