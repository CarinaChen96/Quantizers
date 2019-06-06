# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

import config


def image_show(image, label):
    cv2.imshow(label, image)
    print("Press any key to continue...")
    cv2.waitKey(0)


def image_read():
    im = cv2.imread(config.image_file)
    image_show(im, config.image_file[:-4])
    return im


def transform_image(image, bin_threshold, bin_val):
    flat_image = image.flatten()
    # using bin_val to stand for all the values belongs to that bin
    for i, j in enumerate(flat_image):
        for k in range(1, config.num_levels+1):
            if bin_threshold[k - 1] <= j <= bin_threshold[k]:
                flat_image[i] = bin_val[k-1]
    mean_sq_err = math.sqrt(sum((flat_image-image.ravel())*(flat_image-image.ravel())))
    quantized_image = np.reshape(flat_image, image.shape)
    return mean_sq_err, quantized_image


def lloydMax_quantizer(image):
    flat_image = image.flatten()
    pdf = np.zeros(256, dtype=int)
    for i in flat_image:
        pdf[i] = pdf[i] + 1
    bin_threshold = np.zeros(config.num_levels+1, dtype=float)
    bin_val = np.zeros(config.num_levels, dtype=float)
    q = 256/config.num_levels
    # set the value of each level and the value of q which stand for all the value between every two levels
    for k in range(1, config.num_levels+1):
        bin_threshold[k] = bin_threshold[k-1] + q
        bin_val[k-1] = (bin_threshold[k] + bin_threshold[k-1])/2
    mean_sq_err = 0.0
    prev_mean_sq_err = -0.1
    iteration = 0
    while mean_sq_err > prev_mean_sq_err and iteration < 20:
        prev_mean_sq_err = mean_sq_err
        serr = 0.0
        num = np.zeros(config.num_levels, dtype=int)
        den = np.zeros(config.num_levels, dtype=int)
        for k in range(1, config.num_levels):
            bin_threshold[k] = (bin_val[k-1] + bin_val[k])/2
        for k in range(0, config.num_levels):
            for i in range(int(math.ceil(bin_threshold[k])), int(math.ceil(bin_threshold[k+1]))):
                num[k] = num[k] + i*pdf[i]
                den[k] = den[k] + pdf[i]
            bin_val[k] = num[k]/den[k]
            bin_val[k] = round(bin_val[k])
            for i in range(int(math.ceil(bin_threshold[k])), int(math.floor(bin_threshold[k+1]))):
                serr = serr + (i-bin_val[k])*(i-bin_val[k])*pdf[i]
        mean_sq_err = math.sqrt(serr)
        iteration = iteration + 1
    mean_sq_err, quantized_image = transform_image(image, bin_threshold, bin_val)
    image_show(quantized_image, 'quantized_image: LloydMax Quantizer')
    cv2.imwrite('ll-yourname.png', quantized_image)


def uniform_quantizer(image):
    bin_threshold = np.zeros(config.num_levels+1, dtype=float)
    bin_val = np.zeros(config.num_levels, dtype=float)
    q = 256/config.num_levels
    for k in range(1, config.num_levels+1):
        bin_threshold[k] = bin_threshold[k-1] + q
        bin_val[k-1] = (bin_threshold[k] + bin_threshold[k-1])/2

    # uniform_quantizer is just part of the llmax because it just need to divided into the levels we set
    # and don't need to achieve the smallest mean square quantization error.
    mean_sq_err, quantized_image = transform_image(image, bin_threshold, bin_val)
    image_show(quantized_image, 'quantized_image: Uniform Quantizer')
    cv2.imwrite('uni-yourname.png', quantized_image)


if __name__ == "__main__":
    image = image_read()
    uniform_quantizer(image)
    lloydMax_quantizer(image)
