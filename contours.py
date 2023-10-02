import cv2
import numpy as np

img = cv2.imread('/Users/rpdg/Downloads/photo_2023-06-30 12.43.20.jpeg')
cv2.imshow("Original image", img)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=7., tileGridSize=(8, 8))

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
l, a, b = cv2.split(lab)  # split on 3 different channels

l2 = clahe.apply(l)  # apply CLAHE to the L-channel

lab = cv2.merge((l2, a, b))  # merge channels
img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
cv2.imshow('Increased contrast', img2)

# Выделение контуров

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = 100
ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros(img.shape)
cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
cv2.imshow('contours without contrasts', img_contours)

img_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
thresh = 100
ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros(img.shape)
cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
cv2.imshow('contours with contrast', img_contours)


cv2.waitKey(0)
cv2.destroyAllWindows()
