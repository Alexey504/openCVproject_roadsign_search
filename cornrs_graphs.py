import cv2
import numpy as np
import math
import time


# Вычслить расстояние между точками
def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    l = math.sqrt(
        (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))  # Евклидово расстояние между последней и текущей точкой
    return l


# получить центр масс точек
def get_center(centers, point):
    l = len(centers)
    res = -1
    min_r = float("inf")
    for i in range(0, l):
        center = centers[i]
        x, y, count = center
        r = get_distance(point, (x, y))
        if r >= 10:
            continue
        if r < min_r:
            res = i
            min_r = r
    return res


# Добавить точку в центр масс
def add_to_center(center, point):
    x1, y1, count = center
    count += 1
    x2, y2 = point
    x = x1 + (x2 - x1) / float(count)
    y = y1 + (y2 - y1) / float(count)
    return x, y, count


# путь к указанному входному изображению и
# изображение загружается с помощью команды imread
image = cv2.imread('/Users/rpdg/Downloads/photo_2023-06-30 13.36.32.jpeg')

# конвертировать входное изображение в Цветовое пространство в оттенкахсерого
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
img_blank = np.zeros(image.shape)
img_blank[dest > 0.05 * dest.max()] = [0, 0, 255]

# Сначала создадим список точек
heigh = img_blank.shape[0]
width = img_blank.shape[1]
points = []
for x in range(0, width):
    for y in range(0, heigh):
        if img_blank[y, x, 2] == 255:
            points.append((x, y))

# Теперь будем обрабатывать этот список
points_count = len(points)
print("Количество обрабатываемых точек: ", points_count)
beg_time = time.perf_counter()
centers = []

for i in range(0, points_count):
    point = points[i]
    center_index = get_center(centers, point)
    if center_index == -1:
        x, y = point
        centers.append((x, y, 1))
    else:
        center = centers[center_index]
        centers[center_index] = add_to_center(center, point)
end_time = time.perf_counter()
print("Прошло времени ", end_time - beg_time)

print("Осталось точек ", len(centers))

img_blank1 = np.zeros(image.shape)
for center in centers:
    x, y, count = center
    img_blank1[int(y), int(x), 2] = 255

# окно с выводимым изображением с углами
cv2.imshow('Image with Borders', img_blank1)

# Создание графа

img_blank1 = np.zeros(image.shape)
max = 0
beg_point = None
for center in centers:
    x, y, count = center
    cv2.circle(img_blank1, (int(x), int(y)), 3, (0, 0, 255), 2)
    if count > max:
        max = count
        beg_point = center

centers.remove(beg_point)
graph = []

while len(centers) > 0:
    min = float("inf")
    next_point = None
    for center in centers:
        x1, y1, _ = beg_point
        x2, y2, _ = center
        r = get_distance((x1, y1), (x2, y2))
        if r < min:
            min = r
            next_point = center
    graph.append((beg_point, next_point))
    centers.remove(next_point)
    beg_point = next_point

for edge in graph:
    p1, p2 = edge
    x1, y1, _ = p1
    x2, y2, _ = p2
    cv2.line(img_blank1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=1)

cv2.imshow('Image with Borders Graph', img_blank1)
cv2.imshow('Image', image)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
