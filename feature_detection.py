import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# prepare and read images
img1 = cv.imread('./assets/hp1.jpg')  
img2 = cv.imread('./assets/hp_collection_1.png') 
img3 = cv.imread('./assets/hp1.jpg')

# Convert the images to gray
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#SIFT
sift = cv.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

# Draw the lines to match the feathres
img3 = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()


# Edge Detection
gray = img1
cv.imshow('Gray', gray)

#Blurring
blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# First and Second threshold
canny = cv.Canny(blur, 90, 150)
cv.imshow('Canny Edges', canny)


# # Harris Corner Detection
gray = cv.cvtColor(img3,cv.COLOR_BGR2GRAY)

# find Harris corners
# setting to 32-bit floating point
gray = np.float32(gray)

# image, block size, sobel kernel size, free parameter
dst = cv.cornerHarris(gray, 2, 3, 0.04)

# result are dilated for marking the corners
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img3[dst > 0.01 * dst.max()]=[0, 0, 255]

# image[dest > 0.01 * dest.max()]=[0, 0, 255]

cv.imshow('dst',img3)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

cv.waitKey(0)