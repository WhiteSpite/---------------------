import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle


with open(f'plot_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
data_win = data[0]
data_loss = data[1]
data_draw = data[2]
    
window_size = 15
data_win_smoothed = np.convolve(data_win, np.ones(window_size)/window_size, mode='valid').tolist()
data_loss_smoothed = np.convolve(data_loss, np.ones(window_size)/window_size, mode='valid').tolist()
data_draw_smoothed = np.convolve(data_draw, np.ones(window_size)/window_size, mode='valid').tolist()

max_win_rate = max(data_win_smoothed)
min_draw_rate = min(data_draw_smoothed)
min_loss_rate = min(data_loss_smoothed)

x_smoothed = np.linspace(0, 10000, len(data_win_smoothed)).tolist()

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 10000)
ax.set_ylim(0, 100)
x = np.linspace(0, 10000, 100).tolist()

x_data = [x.pop(0)]
x_data_smoothed = [x_smoothed.pop(0)]

y_data_win = [data_win.pop(0)]
y_data_loss = [data_loss.pop(0)]
y_data_draw = [data_draw.pop(0)]

y_data_win_smoothed = [data_win_smoothed.pop(0)]
y_data_loss_smoothed = [data_loss_smoothed.pop(0)]
y_data_draw_smoothed = [data_draw_smoothed.pop(0)]

line_win, = ax.plot(x_data, y_data_win, c='green', alpha=0.2)
line_loss, = ax.plot(x_data, y_data_loss, c='red', alpha=0.2)
line_draw, = ax.plot(x_data, y_data_draw, c='blue', alpha=0.2)

line_win_smoothed, = ax.plot(x_data_smoothed, y_data_win_smoothed, label='Победы', c='green', linewidth=4)
line_loss_smoothed, = ax.plot(x_data_smoothed, y_data_loss_smoothed, label='Поражения', c='red', linewidth=4)
line_draw_smoothed, = ax.plot(x_data_smoothed, y_data_draw_smoothed, label='Ничьи', c='blue', linewidth=4)

right = 400
annotation_win = ax.annotate('', xy=(x_data_smoothed[-1], y_data_win_smoothed[-1]), xytext=(x_data_smoothed[-1] + right, y_data_win_smoothed[-1]))
annotation_loss = ax.annotate('', xy=(x_data_smoothed[-1], y_data_loss_smoothed[-1]), xytext=(x_data_smoothed[-1] + right, y_data_loss_smoothed[-1]))
annotation_draw = ax.annotate('', xy=(x_data_smoothed[-1], y_data_draw_smoothed[-1]), xytext=(x_data_smoothed[-1] + right, y_data_draw_smoothed[-1]))


def update(frame):
    if x:
        x_data.append(x.pop(0))
        
        y_data_win.append(data_win.pop(0))
        y_data_loss.append(data_loss.pop(0))
        y_data_draw.append(data_draw.pop(0))
        
        line_win.set_data(x_data, y_data_win)
        line_loss.set_data(x_data, y_data_loss) 
        line_draw.set_data(x_data, y_data_draw)
        
    if x_smoothed:    
        x_data_smoothed.append(x_smoothed.pop(0))
        
        y_data_win_smoothed.append(data_win_smoothed.pop(0))
        y_data_loss_smoothed.append(data_loss_smoothed.pop(0))
        y_data_draw_smoothed.append(data_draw_smoothed.pop(0))
        
        line_win_smoothed.set_data(x_data_smoothed, y_data_win_smoothed)
        line_loss_smoothed.set_data(x_data_smoothed, y_data_loss_smoothed)
        line_draw_smoothed.set_data(x_data_smoothed, y_data_draw_smoothed)
        
        global annotation_win, annotation_loss, annotation_draw
        annotation_win.remove()
        annotation_loss.remove()
        annotation_draw.remove()
        
        if y_data_win_smoothed[-1] == max_win_rate:
            ax.scatter(x_data_smoothed[-1], y_data_win_smoothed[-1], c='green', s=100)
            ax.annotate(f'  Max: {str(int(y_data_win_smoothed[-1]))}%', xy=(x_data_smoothed[-1], y_data_win_smoothed[-1]), xytext=(x_data_smoothed[-1] - 450, y_data_win_smoothed[-1] - 9), c='green', fontsize=16)
            globals()['max_win_rate'] = -1
        if y_data_draw_smoothed[-1] == min_draw_rate:
            ax.scatter(x_data_smoothed[-1], y_data_draw_smoothed[-1], c='blue', s=100)
            ax.annotate(f'   Min: {str(int(y_data_draw_smoothed[-1]))}%', xy=(x_data_smoothed[-1], y_data_draw_smoothed[-1]), xytext=(x_data_smoothed[-1] - 500, y_data_draw_smoothed[-1] + 7), c='blue', fontsize=16)
            globals()['min_draw_rate'] = -1
        if y_data_loss_smoothed[-1] == min_loss_rate:
            ax.scatter(x_data_smoothed[-1], y_data_loss_smoothed[-1], c='red', s=100)
            ax.annotate(f'   Min: {str(int(y_data_loss_smoothed[-1]))}%', xy=(x_data_smoothed[-1], y_data_loss_smoothed[-1]), xytext=(x_data_smoothed[-1] - 450, y_data_loss_smoothed[-1] - 8), c='red', fontsize=16)
            globals()['min_loss_rate'] = -1
        
        annotation_win = ax.annotate(str(int(y_data_win_smoothed[-1])) + '%', xy=(x_data_smoothed[-1], y_data_win_smoothed[-1]), xytext=(x_data_smoothed[-1] + right, y_data_win_smoothed[-1]), c='green', fontsize=18)
        annotation_loss = ax.annotate(str(int(y_data_loss_smoothed[-1])) + '%', xy=(x_data_smoothed[-1], y_data_loss_smoothed[-1]), xytext=(x_data_smoothed[-1] + right, y_data_loss_smoothed[-1]), c='red', fontsize=18)
        annotation_draw = ax.annotate(str(int(y_data_draw_smoothed[-1])) + '%', xy=(x_data_smoothed[-1], y_data_draw_smoothed[-1]), xytext=(x_data_smoothed[-1] + right, y_data_draw_smoothed[-1]), c='blue', fontsize=18)
    
    return line_win, line_loss, line_draw, line_win_smoothed, line_loss_smoothed, line_draw_smoothed


animation = FuncAnimation(fig, update, interval=5)

ax.set_xlabel('КОЛИЧЕСТВО ИГР', fontsize=14)
ax.set_ylabel('ПРОЦЕНТ', fontsize=14)
ax.set_xticklabels(map(lambda x: int(x), ax.get_xticks()), fontsize=14)
ax.set_yticklabels(map(lambda x: int(x), ax.get_yticks()), fontsize=14)

fig.suptitle('Статистика обученного MeansQAgent c кластеризованными Q', fontsize=18, y=0.92, weight='bold')

plt.legend(loc='center right', fontsize=14)

plt.show()
