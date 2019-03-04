"""Environment for training the acceleration behavior of vehicles in a loop."""

from flow.envs.base_env import Env
from flow.core import rewards
from flow.core.params import InitialConfig, NetParams, SumoCarFollowingParams
from flow.controllers import IDMController

from gym.spaces.box import Box

import numpy as np

import os
from os.path import expanduser
HOME = expanduser("~")
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='FreeSans', size=12)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shapely.geometry
import itertools


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 11.176,
}


class IntersectionEnv(Env):
    def __init__(self, env_params, sumo_params, scenario):
        print("Starting SoftIntersectionEnv...")
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sumo_params, scenario)

        self.route_idx_table = {
            ('e_1', 'e_4'): 0,
            ('e_1', 'e_6'): 1,
            ('e_1', 'e_8'): 2,
            ('e_3', 'e_2'): 0,
            ('e_3', 'e_8'): 1,
            ('e_3', 'e_6'): 2,
            ('e_5', 'e_8'): 0,
            ('e_5', 'e_2'): 1,
            ('e_5', 'e_4'): 2,
            ('e_7', 'e_6'): 0,
            ('e_7', 'e_4'): 1,
            ('e_7', 'e_2'): 2,
        }

        # setup traffic lights
        tls_list = self.traci_connection.trafficlight.getIDList()
        self.tls_id = tls_list[0]
        tls_definition =\
            self.traci_connection.trafficlight.\
            getCompleteRedYellowGreenDefinition(self.tls_id)
        self.tls_phase = \
            self.traci_connection.trafficlight.getPhase(self.tls_id)
        self.tls_phase_count = 0
        for logic in tls_definition:
            for phase in logic._phases:
                self.tls_phase_count += 1

        # setup reward-related variables
        self.rewards = 0
        self.reward_stats = [0, 0, 0, 0, 0, 0]

        # setup observation-related variables
        self.occupancy_table = np.zeros((16, 5))
        self.speed_table = np.zeros((16, 5))
        self.vehicle_index = {}
        self.vehicle_orient = []

        # setup action-related variables
        self.is_idle = True
        self.miss_pin = False
        self.is_training = False

    # ACTION GOES HERE
    @property
    def action_space(self):
        action = Box(
            low=0,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )
        return action

    def set_action(self, action):
        # if agent_idx == 0:
        #     # pass
        #     tls_phase_increment = int(np.round(
        #         action[1]*self.tls_phase_count
        #     ))
        #     if verbose_mode:
        #         print('Agent index %d and tls phase increment %d' %
        #               (agent_idx, tls_phase_increment))
        #     # self.tls_phase = \
        #     #     self.traci_connection.trafficlight.getPhase(self.tls_id)
        #     # self.tls_phase += tls_phase_increment
        #     # self.tls_phase %= self.tls_phase_count
        #     # self.traci_connection.trafficlight.setPhase(\
        #     #     self.tls_id, self.tls_phase)
        self.is_idle = True
        self.miss_pin = False
        self.is_training = True

        agent_idx = int(np.round(action[0]*30))
        verbose_mode = False
        if agent_idx < 20:
            self.is_idle = False
            if verbose_mode:
                print('Pinning agent %d...' % agent_idx)
                print('Available agents:', self.vehicle_index.keys())
            if agent_idx in self.vehicle_index.keys():
                if verbose_mode:
                    print('Succeded.')
            else:
                if verbose_mode:
                    print('Agent %d not found.' % agent_idx)
                self.miss_pin = True
            if agent_idx in self.vehicle_index.keys():
                veh_list = self.vehicle_index[agent_idx]
                for veh_id in veh_list:
                    try:
                        self.traci_connection.vehicle.setColor(
                            vehID=veh_id, color=(0, 0, 255, 255))
                    except (FatalTraCIError, TraCIException):
                        pass
                    veh_speed = self.vehicles.get_speed(veh_id)
                    next_veh_speed = action[1]*self.scenario.max_speed
                    if verbose_mode:
                        print(
                            'Setting vehicle %s from %f m/s to %f m/s.' %
                            (veh_id, veh_speed, next_veh_speed)
                        )
                    self.traci_connection.vehicle.setSpeed(
                        veh_id, next_veh_speed
                    )
        else:
            if verbose_mode:
                print('Idling...')

    # OBSERVATION GOES HERE
    @property
    def observation_space(self):
        """See class definition."""
        observation = Box(
            low=-np.inf,
            high=np.inf,
            shape=(16, 5, 3),
            dtype=np.float32,)
        return observation

    def get_observation(self, **kwargs):
        #tls_phase = [self.tls_phase]
        #occupancy_table = self.occupancy_table.flatten().tolist()
        #observation = tls_phase + occupancy_table
        #speed_table = self.speed_table.flatten().tolist()
        #observation = tls_phase + speed_table
        return np.dstack((self.occupancy_table, 
                          self.speed_table,
                          self.index_table))

    # REWARD FUNCTION GOES HERE
    def get_reward(self, **kwargs):
        # safety reward
        _sum_collisions = self.sum_collisions * -100
        _pseudo_headway = self.pseudo_headway * -1
        _safety = 0.8 * _sum_collisions + 0.2 * _pseudo_headway
        self.reward_stats[1] += self.sum_collisions
        self.reward_stats[0] += self.pseudo_headway
        
        # performance reward
        _avg_speed = self.avg_speed * 1
        _std_speed = self.std_speed * -1
        _performance = 0.5 * _avg_speed + 0.5 * _std_speed
        self.reward_stats[2] += \
            0 if np.isnan(self.avg_speed) else self.avg_speed
        self.reward_stats[3] += \
            0 if np.isnan(self.std_speed) else self.std_speed

        # consumption reward
        _avg_fuel = self.avg_fuel / self.avg_speed
        _avg_co2 = self.avg_co2 / self.avg_speed
        if np.isnan(_avg_fuel) or np.isinf(_avg_fuel):
            _avg_fuel = 0
        if np.isnan(_avg_co2) or np.isinf(_avg_co2):
            _avg_co2 = 0
        #_cost = 0.5 * _avg_fuel + 0.5 * _avg_co2
        self.reward_stats[4] += _avg_fuel
        self.reward_stats[5] += _avg_co2

        # operation reward
        _operation = 0
        if self.is_idle:
            _operation -= 1
        if self.miss_pin:
            _operation -= 5

        # total reward
        #reward = 0.5 * _safety + 0.4 * _performance + 0.1 * _cost
        # reward = 0.5 * _performance + 0.5 * _cost
        reward = 0.5 * _safety + 0.5 * _performance + _operation
        reward = 0 if np.isnan(reward) else reward

        debug_mode = False
        if debug_mode:
            print("_avg_speed =", _avg_speed)
            print("_std_speed =", _std_speed)
            print("_performance =", _performance)
            print("_avg_fuel =", _avg_fuel)
            print("_avg_co2 =", _avg_co2)
            print("_cost =", _cost)

        return reward

    # UTILITY FUNCTION GOES HERE
    def additional_command(self):
        #self.occupancy_table = np.zeros((16, 5))
        #self.speed_table = np.zeros((16, 5))
        self.occupancy_table = np.zeros((16, 5))
        self.speed_table = np.zeros((16, 5))
        self.index_table = np.zeros((16, 5))
        for row in range(self.index_table.shape[0]):
            for col in range(self.index_table.shape[1]):
                self.index_table[row, col] = \
                    row*self.index_table.shape[1] + col

        self.vehicle_index = {}
        self.vehicle_orient = []
        for veh_id in self.vehicles.get_ids():
            self.vehicle_orient.append(
                self.vehicles.get_orientation(veh_id)
            )
            edge_id = self.traci_connection.vehicle.getRoadID(veh_id)
            if edge_id not in ['e_1', 'e_3', 'e_5', 'e_7']:
                continue
            veh_type = self.traci_connection.vehicle.getTypeID(veh_id)
            route = self.traci_connection.vehicle.getRoute(veh_id)
            edge_idx = int(route[0][2]) // 2
            row_idx = edge_idx * 4
            if veh_type == 'autonomous':
                route_idx = self.route_idx_table[(route[1], route[2])]
                try:
                    self.traci_connection.vehicle.setColor(
                        vehID=veh_id, color=(255, 0, 0, 255))
                except (FatalTraCIError, TraCIException):
                    pass
            else:
                route_idx = 3
            row_idx += route_idx
            col_idx = self.vehicles.get_position(veh_id) // 20
            col_idx = int(min(col_idx, 4))
            try:
                self.occupancy_table[row_idx, col_idx] += 1
                self.speed_table[row_idx, col_idx] += \
                    self.vehicles.get_speed(veh_id)
                agent_idx = int(edge_idx*5 + col_idx)
                if veh_type == 'manned':
                    pass
                elif agent_idx in self.vehicle_index.keys():
                    self.vehicle_index[agent_idx] += [veh_id]
                else:
                    self.vehicle_index[agent_idx] = [veh_id]
            except IndexError:
                raise IndexError(veh_id, veh_type, route,
                      self.vehicles.get_position(veh_id))
        for row_idx in range(self.speed_table[:, :-1].shape[0]):
            for col_idx in range(self.speed_table[:, :-1].shape[1]-1):
                if self.occupancy_table[row_idx, col_idx] != 0:
                    self.speed_table[row_idx, col_idx] /= \
                        self.occupancy_table[row_idx, col_idx]

        # Update key attributes
        self.sum_collisions, self.pseudo_headway = self.compute_collisions()
        speeds = []
        fuels = []
        co2s = []
        for veh_id in self.vehicles.get_ids():
            speeds.append(self.vehicles.get_speed(veh_id))
            fuels.append(self.vehicles.get_fuel(veh_id))
            co2s.append(self.vehicles.get_co2(veh_id))
        self.avg_speed = np.mean(speeds)
        self.std_speed = np.std(speeds)
        self.avg_fuel = np.mean(fuels)
        self.avg_co2 = np.mean(co2s)
        # disable skip to test methods
        self.test_tls(skip=True)
        self.test_reward(skip=True)

        debug_mode = False
        if debug_mode:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(self.speed_table,
                           cmap='jet', vmin=0, vmax=self.scenario.max_speed)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor',
                           labelsize=0)
            ax.set_xticks(np.arange(0,6))
            ax.set_yticks(np.arange(0,16,4))
            ax.set_yticks(np.arange(0,16), minor=True)
            ax.grid(True, which='major')
            ax.grid(True, which='minor', linestyle='--')
            ax.tick_params(which='both', direction='out')
            ax.set_title('time: %d' % self.step_counter)
            fig.colorbar(im, cax=cax, orientation='vertical')
            plt.show()

    # ADDITIONAL HELPER FUNCTIONS
    def compute_collisions(self):
        # TODO: This is currently O(n^2) but can be optimized to O(nlogn).
        polygons = []
        centers = []
        for orient in self.vehicle_orient:
            x, y, ang = orient
            ang = np.radians(ang)
            length, width = 5, 1.8
            _alpha = 0
            pt0 = (x + _alpha*length*np.sin(ang), 
                   y + _alpha*length*np.cos(ang))
            pt00 = (pt0[0] + 0.5*width*np.sin(np.pi/2-ang),
                    pt0[1] - 0.5*width*np.cos(np.pi/2-ang))
            pt01 = (pt0[0] - 0.5*width*np.sin(np.pi/2-ang),
                    pt0[1] + 0.5*width*np.cos(np.pi/2-ang))
            pt1 = (x - (1 - _alpha)*length*np.sin(ang), 
                   y - (1 - _alpha)*length*np.cos(ang))
            pt10 = (pt1[0] + 0.5*width*np.sin(np.pi/2-ang),
                    pt1[1] - 0.5*width*np.cos(np.pi/2-ang))
            pt11 = (pt1[0] - 0.5*width*np.sin(np.pi/2-ang),
                    pt1[1] + 0.5*width*np.cos(np.pi/2-ang))
            polygons.append(
                shapely.geometry.Polygon([pt00, pt01, pt11, pt10]))
            centers.append(np.asarray([x, y]))

        sum_collisions = 0
        for poly1, poly2 in itertools.combinations(polygons, r=2):
            if poly1.intersects(poly2):
                sum_collisions += 1

        pseudo_headway = np.inf
        for center1, center2 in itertools.combinations(centers, r=2):
            distance = np.linalg.norm(center1 - center2)
            if distance < pseudo_headway:
                pseudo_headway = distance

        debug_mode = False
        plot_mode = False
        if debug_mode:
            if sum_collisions > 0:
                #print('Polygons:', polygons)
                print('Sum collisions:', sum_collisions)
                #print('Centers:', centers)
                print('Pseudo headway:', pseudo_headway)
                if plot_mode:
                    fig = plt.figure(figsize=(5,5))
                    ax = fig.add_subplot(111)
                    for poly in polygons:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, color='#6699cc', alpha=0.7,
                            linewidth=3, solid_capstyle='round', zorder=2)
                    ax.set_title('Polygon')
                    ax.axis('equal')
                    ax.set_xlim([100, 140])
                    ax.set_ylim([100, 140])
                    plt.show()

        return sum_collisions, pseudo_headway

    def test_tls(self, skip=True):
        if self.time_counter % 10 == 0 and not skip:
            print("Switching phase...")
            self.tls_phase = np.random.randint(0, self.tls_phase_count-1)
            print("New phase:", self.tls_phase)
            self._set_phase(self.tls_phase)

    def test_reward(self, skip=True):
        if not skip:
            _reward = self.get_reward()
            print('Reward this step:', _reward)
            self.rewards += _reward
            print('Total rewards:', self.rewards)
            print('Cumulative reward stats in log scale:', 
                  np.log(self.reward_stats))

    # DO NOT WORRY ABOUT ANYTHING BELOW THIS LINE >◡<
    def _apply_rl_actions(self, rl_actions):
        self.set_action(rl_actions)

    def get_state(self, **kwargs):
        return self.get_observation(**kwargs)

    def compute_reward(self, actions, **kwargs):
        return self.get_reward(**kwargs)