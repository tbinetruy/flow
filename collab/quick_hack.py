import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='FreeSans', size=12)
from shapely.geometry import LineString, MultiLineString
import pickle

def plot_trajectory(value, ax, type):
    if type in ['east', 'west']:
        idx = 0
    elif type in ['north', 'south']:
        idx = 1
    else:
        raise ValueError

    data = []
    for point in value:
        if type in ['south', 'west']:
            data.append([point[0]/10.0, 240.0 - point[1][idx]])
        else:
            data.append([point[0]/10.0, point[1][idx]])
    data = np.asarray(data)
    if type == 'north':
        ax.plot(data[:,0], data[:,1], 'b')
    elif type == 'south':
        ax.plot(data[:,0], data[:,1], 'c')
    elif type == 'east':
        ax.plot(data[:,0], data[:,1], 'r')
    else:
        ax.plot(data[:,0], data[:,1], 'm')

def add_trajectory(value, trajectories):
    if type in ['east', 'west']:
        idx = 0
    elif type in ['north', 'south']:
        idx = 1
    else:
        raise ValueError

    data = []
    for point in value:
        if type in ['south', 'west']:
            data.append((point[0]/10.0, 240.0 - point[1][idx]))
        else:
            data.append((point[0]/10.0, point[1][idx]))
    #multidata = []
    #for idx in range(len(data)-1):
    #    multidata.append((data[idx], data[idx+1]))
    #trajectories.append(MultiLineString(multidata))
    trajectories.append(LineString(data))

def check_cross(trajectories_A, trajectories_B):
    crosses = []
    for trajectory_a in trajectories_A:
        for trajectory_b in trajectories_B:
            if trajectory_a != trajectory_b:
                #print(trajectory_a)
                #print(trajectory_b)
                cross = trajectory_a.intersection(trajectory_b)
                if not cross.is_empty:
                    print(cross)
                    crosses.append(cross)
    return crosses

def get_intervals(entries, exits):
    intervals = []
    for entry, exit in zip(entries, exits):
        intervals.append([entry.x, exit.x])
    return intervals

def check_overlap(intervals_A, intervals_B):
    overlaps = []
    for interval_a in intervals_A:
        for interval_b in intervals_B:
            if max([interval_a[0], interval_b[0]]) < \
               min([interval_a[1], interval_b[1]]):
                overlaps.append([interval_a, interval_b])
    return overlaps

name = 'trajectory_dict'
with open(name + '.pkl', 'rb') as f:
    trajectory_dict = pickle.load(f)

fig = plt.figure()
ax0 = fig.add_subplot(2,2,1)
ax0.set_title('Eastbound')
ax0.set_xlabel('Time (s)')
ax0.set_ylabel('Distance (m)')
ax0.set_xlim([0, 150])
ax0.set_ylim([20, 220])
ax1 = fig.add_subplot(2,2,2)
ax1.set_title('Southbound')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Distance (m)')
ax1.set_xlim([0, 150])
ax1.set_ylim([20, 220])
ax2 = fig.add_subplot(2,2,3)
ax2.set_title('Westbound')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance (m)')
ax2.set_xlim([0, 150])
ax2.set_ylim([20, 220])
ax3 = fig.add_subplot(2,2,4)
ax3.set_title('Northbound')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Distance (m)')
ax3.set_xlim([0, 150])
ax3.set_ylim([20, 220])

fig = plt.figure()
ax4 = fig.add_subplot(1,1,1)
ax4.axhline(y=112.5)
ax4.axhline(y=127.5)
ax4.set_title('Eastbound')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Distance (m)')
ax4.set_xlim([25, 125])
ax4.set_ylim([112.5, 127.5])

eastbound_trajectories = []
southbound_trajectories = []
westbound_trajectories = []
northbound_trajectories = []

for key, value in trajectory_dict.items():
    #print(key)
    #print(value)
    #ax0.plot()
    if   ('flow_0' in key):
        # Eastbound vehicles
        type = 'east'
        add_trajectory(value, eastbound_trajectories)
        plot_trajectory(value, ax0, type)
        plot_trajectory(value, ax4, type)
    elif ('flow_1' in key):
        # Southbound vehicles
        type = 'south'
        add_trajectory(value, southbound_trajectories)
        plot_trajectory(value, ax1, type)
        plot_trajectory(value, ax4, type)
    elif ('flow_2' in key):
        # Westbound vehicles
        type = 'west'
        add_trajectory(value, westbound_trajectories)
        plot_trajectory(value, ax2, type)
        plot_trajectory(value, ax4, type)
    elif ('flow_3' in key):
        # Northbound vehicles
        type = 'north'
        add_trajectory(value, northbound_trajectories)
        plot_trajectory(value, ax3, type)
        plot_trajectory(value, ax4, type)
    else:
        raise ValueError

# Check illegal pass (a vehicle runs through another vehicle)
conflicts_pass = []
conflicts_pass.append(
    check_cross(eastbound_trajectories, eastbound_trajectories))
conflicts_pass.append(
    check_cross(westbound_trajectories, westbound_trajectories))
conflicts_pass.append(
    check_cross(southbound_trajectories, southbound_trajectories))
conflicts_pass.append(
    check_cross(northbound_trajectories, northbound_trajectories))

entry_line = [LineString([(0, 112.5), (150, 112.5)])]
exit_line = [LineString([(0, 127.5), (150, 127.5)])]
eastbound_entries = []
eastbound_exits = []
southbound_entries = []
southbound_exits = []
westbound_entries = []
westbound_exits = []
northbound_entries = []
northbound_exits = []
eastbound_entries = check_cross(entry_line, eastbound_trajectories)
eastbound_exits = check_cross(exit_line, eastbound_trajectories)
southbound_entries = check_cross(entry_line, southbound_trajectories)
southbound_exits = check_cross(exit_line, southbound_trajectories)
westbound_entries = check_cross(entry_line, westbound_trajectories)
westbound_exits = check_cross(exit_line, westbound_trajectories)
northbound_entries = check_cross(entry_line, northbound_trajectories)
northbound_exits = check_cross(exit_line, northbound_trajectories)

eastbound_intervals = get_intervals(eastbound_entries, eastbound_exits)
southbound_intervals = get_intervals(southbound_entries, southbound_exits)
westbound_intervals = get_intervals(westbound_entries, westbound_exits)
northbound_intervals = get_intervals(northbound_entries, northbound_exits)
conflicts_cross = []
conflicts_cross.append(
    check_overlap(eastbound_intervals, southbound_intervals))
conflicts_cross.append(
    check_overlap(eastbound_intervals, northbound_intervals))
conflicts_cross.append(
    check_overlap(westbound_intervals, southbound_intervals))
conflicts_cross.append(
    check_overlap(westbound_intervals, northbound_intervals))
conflicts_cross.append(
    check_overlap(southbound_intervals, westbound_intervals))
conflicts_cross.append(
    check_overlap(southbound_intervals, eastbound_intervals))
conflicts_cross.append(
    check_overlap(northbound_intervals, westbound_intervals))
conflicts_cross.append(
    check_overlap(northbound_intervals, eastbound_intervals))
for conflicts in conflicts_cross:
    for conflict in conflicts:
        ax4.axvline(x=conflict[0][0], color='k')
        ax4.axvline(x=conflict[0][1], color='k')
        ax4.axvline(x=conflict[1][0], color='k')
        ax4.axvline(x=conflict[1][1], color='k')

plt.tight_layout()
plt.show()
