import cv2
import numpy as np
import math

# my_photo = cv2.imread('/Users/rpdg/Downloads/photo_2023-07-02 13.49.16.jpeg')
my_photo = cv2.imread('/Users/rpdg/Downloads/photo_2023-07-02 13.49.19.jpeg')

# --------------------- предобработка изображения --------------
filterd_image = cv2.medianBlur(my_photo, 5)
# filterd_image = cv2.GaussianBlur(my_photo, (5, 5), 0)

# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# img = cv2.filter2D(filterd_image, -1, kernel)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(12, 12))
# lab = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
# l, a, b = cv2.split(lab)  # split on 3 different channels
# l2 = clahe.apply(l)  # apply CLAHE to the L-channel
# lab = cv2.merge((l2, a, b))  # merge channels
# img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

img_grey = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)

cv2.imshow('blur', img_grey)
# ----------------------------------------------------------------

# --------------------- выделение контуров -------------------------
thresh = 100
ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# img_contours = np.uint8(np.zeros((my_photo.shape[3], my_photo.shape[1])))
img_contours = np.uint8(np.zeros((my_photo.shape[0], my_photo.shape[1])))

cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)

cv2.imshow('res', img_contours)
# -------------------------------------------------------------------


# --------------- работа сотдельными контурами -------------------------
# sel_countours = []
# sel_countours.append(contours[3])
# sel_countours.append(contours[7])
# sel_countours.append(contours[8])
# cv2.drawContours(img_contours, sel_countours, -1, (255, 255, 255), 1)

# самый большой контур
# max = 0
# sel_countour = None
# for countour in contours:
#     if countour.shape[0] > max:
#         sel_countour = countour
#         max = countour.shape[0]

# for point in sel_countour:
#     y=int(point[0][1])
#     x=int(point[0][0])
#     img_contours[y,x]=255

# cv2.drawContours(img_contours, [sel_countour], -1, (255, 255, 255), 1)
# cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
# -------------------------------------------------------------------------

# ----------------------- аппроксимация контура --------------------------
def custom_sort(countour):
    return -countour.shape[0]


contours = list(contours)
contours.sort(key=custom_sort)
sel_countour = contours[0]

# calc arclentgh
arclen = cv2.arcLength(sel_countour, True)
# print(arclen)

# do approx
eps = 0.003
epsilon = arclen * eps
approx = cv2.approxPolyDP(sel_countour, epsilon, True)
print(len(approx))
# -------------------------------------------------------------------------


# поиск центра контура
sum_x = 0.0
sum_y = 0.0
for point in approx:
    x = float(point[0][0])
    y = float(point[0][1])
    sum_x += x
    sum_y += y
xc = sum_x / float(len(approx))
yc = sum_y / float(len(approx))

# поиск точки наиболее удаленной от центра
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


def polar_sort(item):
    return item[0][0]


def get_polar_coordinates(x0, y0, x, y, xc, yc):
    """Функция дляя вычисления полярных координат"""
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
    angle = math.acos(cos_angle)
    if sgn < 0:
        angle = 2 * math.pi - angle
    return angle, r


# нахождение поляярных координат
polar_coordinates = []
x0 = approx[beg_point][0][0]
y0 = approx[beg_point][0][1]
print(x0, y0)
for point in approx:
    x = int(point[0][0])
    y = int(point[0][1])
    angle, r = get_polar_coordinates(x0, y0, x, y, xc, yc)
    polar_coordinates.append(((angle, r), (x, y)))
print(polar_coordinates)
polar_coordinates.sort(key=polar_sort)

# draw the result
canvas = my_photo.copy()
for pt in approx:
    cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)

cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 2, cv2.LINE_AA)


def get_cos_edges(edges):
    """функция вычисления косинуса угла между гранями """
    dx1, dy1, dx2, dy2 = edges
    r1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    r2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
    return (dx1 * dx2 + dy1 * dy2) / r1 / r2


def get_coords(item1, item2, item3):
    """Функцция вычесленияя относительных координат"""
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


img_contours = np.uint8(np.zeros((my_photo.shape[0], my_photo.shape[1])))
# отображение полярных координат
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

# инвариантное описание
coses = []
coses.append(get_cos_edges(get_coords(polar_coordinates[size - 1], polar_coordinates[0], polar_coordinates[1])))
for i in range(1, size - 1):
    coses.append(get_cos_edges(get_coords(polar_coordinates[i - 1], polar_coordinates[i], polar_coordinates[i + 1])))
coses.append(get_cos_edges(get_coords(polar_coordinates[size - 2], polar_coordinates[size - 1], polar_coordinates[0])))
print("инвариант", coses)


# ------------------------ отображение ----------------------------
cv2.line(img_contours, (x1, y1), (x2, y2), 255, thickness=size)
# отображение точки наиболее удаленной от центра
point = approx[beg_point]
x = float(point[0][0])
y = float(point[0][1])
cv2.circle(img_contours, (int(x), int(y)), 7, (255, 255, 255), 2)

cv2.circle(img_contours, (int(xc), int(yc)), 7, (255, 255, 255), 2)  # отображение центра
cv2.drawContours(img_contours, [approx], -1, (255, 255, 255), 1)

cv2.imshow('origin_aprox', canvas)
cv2.imshow('res_aprox', img_contours)
# -------------------------------------------------------------------------
cv2.waitKey()
cv2.destroyAllWindows()
