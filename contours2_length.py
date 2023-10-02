import cv2
import numpy as np
import math
import os

img = cv2.imread("/Users/rpdg/Downloads/photo_2023-07-02 13.49.16.jpeg")
filterd_image = cv2.medianBlur(img, 5)
gray = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)
thresh = 100


def custom_sort(countour):
    return -countour.shape[0]


def polar_sort(item):
    return item[0][0]


def get_normalized_vector(list):
    arr = np.array(list)
    return (arr - arr.min()) / (arr.max() - arr.min())


# Эвклидово расстояние между двумя элементами
def get_length(item1, item2):
    _, point1 = item1
    _, point2 = item2
    x1, y1 = point1
    x2, y2 = point2
    dx = x1 - x2
    dy = y1 - y2
    r = math.sqrt(dx * dx + dy * dy)
    return r


def get_cos_edges(edges):
    dx1, dy1, dx2, dy2 = edges
    r1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    r2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
    return (dx1 * dx2 + dy1 * dy2) / r1 / r2


def get_polar_coordinates(x0, y0, x, y, xc, yc):
    # Первая координата в полярных координатах - радиус
    dx = xc - x
    dy = yc - y
    r = math.sqrt(dx * dx + dy * dy)

    # Вторая координата в полярных координатах - узел, вычислим относительно начальной точки
    dx0 = xc - x0
    dy0 = yc - y0
    r0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
    scal_mul = dx0 * dx + dy0 * dy
    cos_angle = scal_mul / r / r0
    sgn = dx0 * dy - dx * dy0  # опредедляем, в какую сторону повернут вектор
    if cos_angle > 1:
        if cos_angle > 1.0001:
            raise Exception("Что-то пошло не так")
        cos_angle = 1
    angle = math.acos(cos_angle)
    if sgn < 0:
        angle = 2 * math.pi - angle
    return angle, r


def get_coords(item1, item2, item3):
    _, point1 = item1
    _, point2 = item2
    _, point3 = item3
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    dx1 = x1 - x2
    dy1 = y1 - y2
    dx2 = x3 - x2
    dy2 = y3 - y2
    return dx1, dy1, dx2, dy2


# get threshold image
ret, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

# find contours without approx
contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = list(contours)
contours.sort(key=custom_sort)
sel_countour = contours[0]

# calc arclentgh
arclen = cv2.arcLength(sel_countour, True)

# do approx
eps = 0.003
epsilon = arclen * eps
approx = cv2.approxPolyDP(sel_countour, epsilon, True)

sum_x = 0.0
sum_y = 0.0
for point in approx:
    x = float(point[0][0])
    y = float(point[0][1])
    sum_x += x
    sum_y += y
xc = sum_x / float(len((approx)))
yc = sum_y / float(len((approx)))

max = 0
beg_point = -1
for i in range(0, len(approx)):
    point = approx[i]
    x = float(point[0][0])
    y = float(point[0][1])
    dx = x - xc
    dy = y - yc
    r = math.sqrt(dx * dx + dy * dy)
    if r > max:
        max = r
        beg_point = i

polar_coordinates = []
x0 = approx[beg_point][0][0]
y0 = approx[beg_point][0][1]

for point in approx:
    x = int(point[0][0])
    y = int(point[0][1])
    angle, r = get_polar_coordinates(x0, y0, x, y, xc, yc)
    polar_coordinates.append(((angle, r), (x, y)))

polar_coordinates.sort(key=polar_sort)

img_contours = np.uint8(np.zeros((img.shape[0], img.shape[1])))
size = len(polar_coordinates)
for i in range(1, size):
    _, point1 = polar_coordinates[i - 1]
    _, point2 = polar_coordinates[i]
    x1, y1 = point1
    x2, y2 = point2
    cv2.line(img_contours, (x1, y1), (x2, y2), 255, thickness=i)
_, point1 = polar_coordinates[size - 1]
_, point2 = polar_coordinates[0]
x1, y1 = point1
x2, y2 = point2
cv2.line(img_contours, (x1, y1), (x2, y2), 255, thickness=size)

cv2.circle(img_contours, (int(xc), int(yc)), 7, (255, 255, 255), 2)

coses = []
coses.append(get_cos_edges(get_coords(polar_coordinates[size - 1], polar_coordinates[0], polar_coordinates[1])))
for i in range(1, size - 1):
    coses.append(get_cos_edges(get_coords(polar_coordinates[i - 1], polar_coordinates[i], polar_coordinates[i + 1])))
coses.append(get_cos_edges(get_coords(polar_coordinates[size - 2], polar_coordinates[size - 1], polar_coordinates[0])))
print(coses)

lengths = []
for i in range(size - 1):
    lengths.append(get_length(polar_coordinates[i], polar_coordinates[i + 1]))
lengths.append(get_length(polar_coordinates[size - 1], polar_coordinates[0]))
print(get_normalized_vector(lengths))

point = approx[beg_point]
x = float(point[0][0])
y = float(point[0][1])
cv2.circle(img_contours, (int(x), int(y)), 7, (255, 255, 255), 2)

cv2.imshow('origin', img)  # выводим итоговое изображение в окно
cv2.imshow('res', img_contours)  # выводим итоговое изображение в окно

cv2.waitKey()
cv2.destroyAllWindows()
