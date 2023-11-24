import cv2 as cv
import numpy as np
import sys
import os
import time
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt


def cal_BSR(cxr_path, gt_path, bs_path):
    cxr = cv.imread(cxr_path)
    gt = cv.imread(gt_path)
    bs = cv.imread(bs_path)

    cxr = cxr / 255
    gt = gt / 255
    bs = bs / 255
    bone = cv.subtract(cxr, gt)

    gt = cv.resize(gt, (256, 256))
    bs = cv.resize(bs, (256, 256))
    bone = cv.resize(bone, (256, 256))

    bs += np.average(cv.subtract(gt, bs))

    bias = cv.subtract(gt, bs)
    bias[bias < 0] = 0

    BSR = 1 - np.sum(bias ** 2) / np.sum(bone ** 2)
    return BSR


def cal_MSE(gt_path, bs_path):
    gt = cv.imread(gt_path)
    bs = cv.imread(bs_path)

    gt = gt / 255
    bs = bs / 255

    gt = cv.resize(gt, (256, 256))
    bs = cv.resize(bs, (256, 256))

    MSE = np.mean((gt - bs) ** 2)
    return MSE


def cal_SSIM(gt_path, bs_path):
    gt = cv.imread(gt_path)
    bs = cv.imread(bs_path)

    gt = gt / 255
    bs = bs / 255

    gt = cv.resize(gt, (256, 256))
    bs = cv.resize(bs, (256, 256))

    SSIM = ssim(gt, bs, channel_axis=2, data_range=1)
    return SSIM


def cal_PSNR(gt_path, bs_path):
    gt = cv.imread(gt_path)
    bs = cv.imread(bs_path)

    gt = cv.resize(gt, (256, 256))
    bs = cv.resize(bs, (256, 256))

    mse = np.mean((gt - bs) ** 2)

    if (mse == 0):
        return 100
    max_pixel = 255.0

    PSNR = 20 * log10(max_pixel / sqrt(mse))
    return PSNR


if __name__ == "__main__":
    CXR_path = "./A"
    GT_path = "./C"
    BS_path = "./internal_test_suppressed_10KFold217_uncropped"

    BSR_list = []
    MSE_list = []
    SSIM_list = []
    PSNR_list = []

    for filename in os.listdir(BS_path):
        cxr_path = os.path.join(CXR_path, filename)
        gt_path = os.path.join(GT_path, filename)
        bs_path = os.path.join(BS_path, filename)

        BSR = cal_BSR(cxr_path, gt_path, bs_path)
        MSE = cal_MSE(gt_path, bs_path)
        SSIM = cal_SSIM(gt_path, bs_path)
        PSNR = cal_PSNR(gt_path, bs_path)

        BSR_list.append(BSR)
        MSE_list.append(MSE)
        SSIM_list.append(SSIM)
        PSNR_list.append(PSNR)

        print(f"{filename} BSR: {BSR} MSE: {MSE} SSIM:{SSIM} PSNR:{PSNR}")

    print("Average BSR:", sum(BSR_list) / len(BSR_list))
    print("Average MSE:", sum(MSE_list) / len(MSE_list))
    print("Average SSIM:", sum(SSIM_list) / len(SSIM_list))
    print("Average PSNR:", sum(PSNR_list) / len(PSNR_list))
