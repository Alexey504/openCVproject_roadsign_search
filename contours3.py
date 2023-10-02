import cv2
import numpy as np
import math

template_vector = np.array([0.09975077, 1, 0.01461883, 0, 0.01617952, 0.99030721, 0.09387105])
distance_thresh = 0.1

img = cv2.imread("/Users/rpdg/Downloads/photo_2023-07-02 13.49.21.jpeg")
filterd_image = cv2.medianBlur(img, 5)
gray = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)
thresh = 100


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


# get threshold image
ret, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

# find contours without approx
contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for sel_countour in contours:
    # calc arclentgh
    arclen = cv2.arcLength(sel_countour, True)

    # do approx
    eps = 0.003
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(sel_countour, epsilon, True)

    # Обрабатываем только контуры длиной 4 углов
    if len(approx) == 7:

        # вычислим центр тяжести контура
        sum_x = 0.0
        sum_y = 0.0
        for point in approx:
            x = float(point[0][0])
            y = float(point[0][1])
            sum_x += x
            sum_y += y
        xc = sum_x / float(len(approx))
        yc = sum_y / float(len(approx))

        # найдем начальную точку
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

        # Вычислми полярные координаты
        polar_coordinates = []
        x0 = approx[beg_point][0][0]
        y0 = approx[beg_point][0][1]
        for point in approx:
            x = int(point[0][0])
            y = int(point[0][1])
            angle, r = get_polar_coordinates(x0, y0, x, y, xc, yc)
            polar_coordinates.append(((angle, r), (x, y)))

        # Создадим вектор описание
        polar_coordinates.sort(key=polar_sort)
        size = len(polar_coordinates)
        lengths = []
        for i in range(size - 1):
            lengths.append(get_length(polar_coordinates[i], polar_coordinates[i + 1]))
        lengths.append(get_length(polar_coordinates[size - 1], polar_coordinates[0]))
        descr = get_normalized_vector(lengths)

        # Вычислим эвклидово расстояние
        square = np.square(descr - template_vector)
        sum_square = np.sum(square)
        distance = np.sqrt(sum_square)
        if distance < distance_thresh:
            for i in range(1, size):
                _, point1 = polar_coordinates[i - 1]
                _, point2 = polar_coordinates[i]
                x1, y1 = point1
                x2, y2 = point2
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)
            _, point1 = polar_coordinates[size - 1]
            _, point2 = polar_coordinates[0]
            x1, y1 = point1
            x2, y2 = point2
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=size)

cv2.imshow('origin', img)  # выводим итоговое изображение в окно

cv2.waitKey()
cv2.destroyAllWindows()
