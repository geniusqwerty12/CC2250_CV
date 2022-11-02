import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./assets/beach.jpg')
img1 = cv.imread("./assets/dog_and_cat2.jpg")
img2 = cv.imread('./assets/machine_learning.jpg')

# image -> matrix -> image
# img_matrix = cv.imread("./assets/beach.jpg")
# img2_matrix = cv.imread("./assets/machine_learning.jpg")
# # print(img_matrix.shape)
# img_add = np.array(img_matrix + img2_matrix)
# # print(img_add)
# img_total = Image.fromarray(img_add)
# img_total.show()

# brightness
# img_matrix = cv.imread("./assets/beach.jpg", 0)
# cv.imshow("original image",img_matrix)
# img_brightness = np.array(img_matrix) - 80
# # print(img_add)
# img_total = Image.fromarray(img_brightness)
# img_total.show()

# Image blending
# img3 = cv.addWeighted(img, 0.7, img2, 0.3, 0)
# cv.imshow('new image', img3)

# image histogram equalization
# img1 = cv.imread("./assets/dog_and_cat2.jpg", 0)

# # # histogram
# gray_hist = cv.calcHist([img1], [0], None, [256], [0,256] )

# # # Grayscale Histogram
# plt.figure()
# plt.title('Histogram 1')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

# # # histogram equalization 
# equ = cv.equalizeHist(img1)

# # # image histogram
# gray_hist2 = cv.calcHist([equ], [0], None, [256], [0,256] )

# # # Grayscale Histogram
# plt.figure()
# plt.title('Histogram 2')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist2)
# plt.xlim([0,256])
# plt.show()


# cv.imshow('Dog and Cat', img1)
# cv.imshow('Image with Equalized Histogram', equ)

# # BLUR
# cv.imshow('Original Image', img1)

# # Averaging
# average = cv.blur(img1, (7,7))
# cv.imshow('Average Blur', average)

# Gaussian Blur
# gauss = cv.GaussianBlur(img1, (7,7), 0)
# cv.imshow('Gaussian Blur', gauss)

# # # Median Blur
# median = cv.medianBlur(img1, 7)
# cv.imshow('Median Blur', median)

# IMAGE THRESHOLDING
# gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# # Simple Thresholding Binary
# threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY )
# cv.imshow('Simple Thresholded', thresh)

# threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV )
# cv.imshow('Simple Thresholded Inverse', thresh_inv)

# # Unsharp masking
cv.imshow('Original', img1)
gaussian_3 = cv.GaussianBlur(img1, (5, 5), 2.0)
cv.imshow('Blur', gaussian_3)
unsharp_image = cv.addWeighted(img1, 2.0, gaussian_3, -1.0, 0)
cv.imshow('Unsharped', unsharp_image)

cv.waitKey(0)
