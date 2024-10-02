import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def round_off_rating(number):
    return round(number * 2) / 2


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def back_ident(img_path, n_std):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hist = np.histogram(img, 255, (0, 255))
    bins = hist[1][1:] - 0.5
    vals = hist[0] * 1 / (img.shape[0] * img.shape[1])

    max_loc1 = np.where(vals == max(vals))[0]
    max_val = bins[max_loc1] + 0.5

    def known_function(x, b):
        return (1 / (b * np.sqrt(2 * np.pi))) * np.exp(-(x - max_val) ** 2 / (2 * (b ** 2)))

    popt, pcov = curve_fit(known_function, bins, vals)

    bin_range = bins[find_nearest(bins, round_off_rating((max_val - n_std * popt[0])[0])):
                     find_nearest(bins, round_off_rating((max_val + n_std * popt[0])[0]))]

    img1 = img
    for k in bin_range:
        img1[img1 == k.astype('uint8')] = 0

    return img1


# Specify input directory and image name
input_dir = "//Users/vnpawan/Documents/Output_SAM_comp input 2"
i = 'images_00114.tiff'

image_path = os.path.join(input_dir, i)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# The pixel intensity histogram is considered to be a normal distribution with mean identified as the peak of the histogram.
# This assumption is true for most microscope images as the majority of the image is usually background noise.
# n is the multiplier for standard deviation. The range of pixels considered as background increases with increasing value of n.
# General values to try for n are [0.1, 0.2, 0.25, 0.5, 0.75, 1].

n = 1

back_img = back_ident(os.path.join(input_dir, i), n)

plt.figure()
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(back_img, cmap='gray')
plt.axis('off')
plt.title('BG identified')

plt.show(block=True)
