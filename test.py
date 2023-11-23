import numpy as np  
from timer import timer
import matplotlib.pyplot as plt

# Ваш массив точек
array = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

# Разделение массива на x и y координаты
x_values = [point[0] for point in array]
y_values = [point[1] for point in array]

# Создание точечного графика
plt.scatter(x_values, y_values)

# Добавление меток к осям
plt.xlabel('X')
plt.ylabel('Y')

# Добавление заголовка графика
plt.title('Точечный график')

# Показать график
plt.show()