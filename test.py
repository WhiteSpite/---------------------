import numpy as np  
from timer import timer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import pickle

num_proups = 149

def find_group_boundaries(array, num_groups=num_proups):
    X = np.array(array).reshape(-1, 1)
    model = DBSCAN(eps=200, min_samples=1)  # Подберите параметры eps и min_samples под вашу задачу
    model.fit(X)

    labels = model.labels_
    clusters_num = len(set(labels))
    # for i in labels:
    #     print(i)
    # exit()
    # idx = np.argsort()
    # lut = np.zeros_like(idx)
    # lut[idx] = np.arange(num_groups)
    # labels = lut[labels]
    
    
    return labels, clusters_num


with open('X_q_table.pkl', 'rb') as f:
                q_table = pickle.load(f)
 
array = []               
for hash in q_table:
    if type(q_table[hash]) is not bool:
        for action in q_table[hash]:
            # array.append(q_table[hash][action]['win'] / (q_table[hash][action]['draw'] + q_table[hash][action]['loss']))
            array.append((q_table[hash][action]['win'] - q_table[hash][action]['draw']/8) - q_table[hash][action]['loss'])

# Пример использования

labels, clasters_num = find_group_boundaries(array)
print(clasters_num)
lists = [[] for i in range(clasters_num)]
for val, label in zip(array, labels):
    lists[label].append(val)

x_values = array
y_values = [0 for i in range(len(array))]

colors = []
for i in range(clasters_num//2):
    colors.extend(["red", "blue", "green"])

for list in lists:
    plt.scatter([list], [0 for i in range(len(list))], s=0.5, c=colors[lists.index(list)])


# Добавление меток к осям
plt.xlabel('X')
plt.ylabel('Y')

# Добавление заголовка графика
plt.title('Точечный график')

# Показать график
plt.show()