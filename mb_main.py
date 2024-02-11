import cv2
import numpy as np
import sqlite3 as sql
import json


def data_create(img, img_clear):
    # Загрузка изображения
    image = img.copy()

    # Список уже нарисованных боксов
    drawn_boxes = []

    # Конвертирование изображения в HSV цветовое пространство
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определение диапазона зелёного цвета
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Маска для выделения зелёных пикселей
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Закрытие операция для устранения шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Нахождение контуров
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img_clear.copy()

    # Предполагаем, что petri_dish_diameter задано в миллиметрах
    petri_dish_diameter = 90  # Например, 90 мм
    image_width, image_height, _ = image.shape

    for i, cnt in enumerate(contours):
        # Вычисление ограничивающего прямоугольника
        x, y, w, h = cv2.boundingRect(cnt)

        # Расчёт диаметра окружности, описанной вокруг объекта (в пикселях)
        diameter_outside_pixels = cv2.arcLength(cnt, True) * 2

        # Расчёт диаметра окружности, вписанной в объект (в пикселях)
        _, diameter_inside_pixels = cv2.minEnclosingCircle(cnt)
        diameter_inside_pixels *= 2

        # Преобразование диаметра объекта из пикселей в миллиметры
        diameter_outside_mm = diameter_outside_pixels * petri_dish_diameter / min(image_width, image_height)
        diameter_inside_mm = diameter_inside_pixels * petri_dish_diameter / min(image_width, image_height)

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # Добавление ID объекта в левый верхний угол изображения
        cv2.putText(output, str(i), (x + 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Добавление бокса в список уже нарисованных боксов
        # drawn_boxes.append((x, y, w, h))

        # |||||Вычисление среднего цвета объекта|||||
        # Создание маски для выделения объекта без зелёного контура
        obj_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(obj_mask, contours, i, 255, -1)
        obj_mask = cv2.bitwise_not(obj_mask)

        # Вычисление среднего цвета объекта
        mean_color = cv2.mean(image, mask=obj_mask)[:3]
        mean_color_int = map(int, mean_color)
        mean_color_str = ', '.join(map(str, mean_color_int))

        with sql.connect('Data.sqlite') as db:
            cursor = db.cursor()
            cursor.execute(
                """INSERT INTO data (id, weight, hight, color) VALUES (?, ?, ?, ?)""",
                (i, float(f'{diameter_outside_mm:.2f}'), float(f'{diameter_inside_mm:.2f}'), mean_color_str)
            )
            db.commit()
    # Сохранение вырезанных объектов как отдельные изображения
    #     cv2.imwrite(f'object_{i}.png', output)

    # Отображение результата
    cv2.imshow('Output', output)
    print(*drawn_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
