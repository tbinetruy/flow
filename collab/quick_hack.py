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

def check_intersections(trajectories_A, trajectories_B):
    intersects = []
    for trajectory_a in trajectories_A:
        for trajectory_b in trajectories_B:
            if trajectory_a != trajectory_b:
                #print(trajectory_a)
                #print(trajectory_b)
                intersect = trajectory_a.intersection(trajectory_b)
                if not intersect.is_empty:
                    print(intersect)
                    intersects.append(intersect)
    return intersects

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

# Check if a vehicle runs through another vehicle
conflicts = []
conflicts.append(check_intersections(
    eastbound_trajectories, eastbound_trajectories))
conflicts.append(check_intersections(
    westbound_trajectories, westbound_trajectories))
conflicts.append(check_intersections(
    southbound_trajectories, southbound_trajectories))
conflicts.append(check_intersections(
    northbound_trajectories, northbound_trajectories))
print(conflicts)

plt.tight_layout()
plt.show()
