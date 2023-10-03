import cv2
import numpy as np
import math

# путь к указанному входному изображению и
# изображение загружается с помощью команды imread
image = cv2.imread('/Users/rpdg/Downloads/photo_2023-06-30 13.36.32.jpeg')

# clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
# lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
# l, a, b = cv2.split(lab)  # split on 3 different channels
# l2 = clahe.apply(l)  # apply CLAHE to the L-channel
# lab = cv2.merge((l2, a, b))  # merge channels
# img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
# cv2.imshow('Increased contrast', img2)

# конвертировать входное изображение в Цветовое пространство в оттенках серого
operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# изменить тип данных
# установка 32-битной плавающей запятой
operatedImage = np.float32(operatedImage)

# применить метод cv2.cornerHarris
# для определения углов с соответствующими
# значения в качестве входных параметров
dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

# Результаты отмечены через расширенные углы
dest = cv2.dilate(dest, None)

# Возвращаясь к исходному изображению,
# с оптимальным пороговым значением
image_blank = np.zeros(image.shape)
image_blank[dest > 0.05 * dest.max()] = [0, 0, 255]

# окно с выводимым изображением с углами
cv2.imshow('Image with Borders 1', image_blank)


# heigh = image_blank.shape[0]
# width = image_blank.shape[1]
# count = 0
# for x in range(width):
#     for y in range(heigh):
#         if image_blank[y, x, 2] == 255:
#             print(x, y)
#             count += 1
# print("Всего", count, "точек")

# Получить последнюю точку в списке списоков и одиночных точек
def get_last_point(ls):
    lls = len(ls)
    if lls > 0:
        item = ls[lls - 1]
        if type(item) == list:
            if len(item) > 0:
                x, y = item[len(item) - 1]
            else:
                return 0, 0, False
        else:
            x, y = item
        return x, y, True
    return 0, 0, False


# Вычслить расстояние между точками
def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    l = math.sqrt(
        (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))  # Евклидово расстояние между последней и текущей точкой
    return l


# Добавить точку в кластер
def add_point_to_claster(x, y, ls):
    lls = len(ls)
    item = ls[lls - 1]
    if type(item) == list:
        item.append((x, y))
    else:
        x1, y1 = item
        item = [(x1, y1)]
        item.append((x, y))
        ls[lls - 1] = item


def calk_center(ls):
    ix = 0
    iy = 0
    l = float(len(ls))
    for point in ls:
        x, y = point
        ix += x
        iy += y
    return round(ix / l), round(iy / l)


heigh = image_blank.shape[0]
width = image_blank.shape[1]
points = []
for x in range(0, width):
    for y in range(0, heigh):
        if image_blank[y, x, 2] == 255:
            x1, y1, point_is = get_last_point(points)
            if point_is:
                l = get_distance((x1, y1), (x, y))
                if l < 3:
                    add_point_to_claster(x, y, points)
                    continue
            points.append((x, y))
centers = []
for point in points:
    if type(point) == list:
        centers.append(calk_center(point))
    else:
        centers.append(point)

image_blank1 = np.zeros(image.shape)
for point in centers:
    print(point)
    x, y = point
    image_blank1[y, x, 2] = 255

# окно с выводимым изображением с углами
cv2.imshow('Image with Borders', image_blank1)
cv2.imshow('Image', image)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
