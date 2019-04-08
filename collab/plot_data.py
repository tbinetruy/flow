import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import matplotlib
import matplotlib.cm as cm
from matplotlib import collections
matplotlib.rc('font', family='FreeSans', size=14)
from copy import copy
import sys

input = sys.argv[1]
data = np.load('%s.npy' % input)
data = data[0::10,:,:]
print('Processing %s.npy of shape' % input, data.shape)

network = np.load('network.npy')
for idx, line in enumerate(network):
    _line = []
    for x, y in zip(line[0::2], line[1::2]):
        _line.append((x, y))
    network[idx] = _line

network_lc = collections.LineCollection(
    network, linewidths=1, colors='w')

T = data.shape[0]
max_x = data[:, :, 0].max()
min_x = data[:, :, 0].min()
max_y = data[:, :, 1].max()
min_y = data[:, :, 1].min()
max_speed = data[:, :, 2].max()
min_speed = data[:, :, 2].min()
mean_speeds = []

for t in range(T):
    plt.figure(figsize=(9, 12))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax1.add_collection(copy(network_lc))
    ax1.scatter(data[t, :, 0], data[t, :, 1], alpha=1, 
                color=cm.RdYlGn((data[t, :, 2]-min_speed)/(max_speed-min_speed)))
    ax1.set_xlim([min_x-10, max_x+10])
    ax1.set_ylim([min_y-10, max_y+10])
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')

    ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    mean_speed = data[t, :, 2].mean()
    overall_mean_speed = data[:t+1, :, 2].mean()
    mean_speeds.append(mean_speed)
    _mean_speeds = np.array(mean_speeds)
    ax2.scatter(np.arange(t+1), _mean_speeds, alpha=1, 
                color=cm.RdYlGn((_mean_speeds-min_speed)/(max_speed-min_speed)))
    ax2.axhline(y=overall_mean_speed)
    ax2.text(T-175, overall_mean_speed+0.5, 'Average Speed')
    ax2.axvline(x=t)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('System Speed (m/s)')
    ax2.set_xlim([-10, T+10])
    ax2.set_ylim([-1, 8])

    plt.tight_layout()
    plt.savefig('%s_demo/%03d.png' % (input, t), bbox_inches='tight')
    plt.close()
