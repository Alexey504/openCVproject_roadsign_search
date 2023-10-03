import cv2
import numpy as np


my_photo = cv2.imread('/Users/rpdg/Downloads/scale_1200-2.jpeg')
# my_photo = cv2.imread('/Users/rpdg/Downloads/63456.pv7boc.1280.jpg')
# my_photo = cv2.imread('/Users/rpdg/Downloads/_dsc0103_w650.jpg')


# поиск по красному цвету
filtered_image = cv2.medianBlur(my_photo, 7)

img = cv2.cvtColor(my_photo, cv2.COLOR_BGR2HSV)

h_channel = my_photo[:, :, 0]
v_channel = my_photo[:, :, 2]
bin_img = np.zeros(my_photo.shape)
bin_img[(h_channel < 100) * (h_channel > 40) * (v_channel > 80)] = [0, 0, 255]
kernel = np.ones((5, 5), np.uint8)
# opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('h_channel', h_channel)
cv2.imshow('v_channel', v_channel)
cv2.imshow('result', closing)

# поиск по контуру
img_grey = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
thresh = 100
ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img_contours = np.zeros(img.shape)
img_contours = np.uint8(np.zeros((my_photo.shape[0], my_photo.shape[1])))
cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
cv2.imshow('contours', img_contours)


# поиск по окружности
rows = img_grey.shape[0]
circles = cv2.HoughCircles(img_grey, cv2.HOUGH_GRADIENT, 1, rows / 8,
                           param1=210, param2=80,
                           minRadius=30, maxRadius=300)

res = np.zeros(my_photo.shape)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(res, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(res, center, radius, (255, 0, 255), 3)
        cv2.circle(my_photo, center, radius, (255, 0, 255), 3)

cv2.imshow('origin', my_photo)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
