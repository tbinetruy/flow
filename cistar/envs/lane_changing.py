from cistar.envs.base_env import SumoEnvironment
from cistar.core import rewards
from cistar.controllers.car_following_models import *

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
import numpy as np
from numpy.random import normal


class SimpleLaneChangingAccelerationEnvironment(SumoEnvironment):
    """
    Fully functional environment. Takes in an *acceleration* as an action. Reward function is negative norm of the
    difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
    velocities for each vehicle.
    """

    @property
    def action_space(self):
        """
        Actions are:
         - a (continuous) acceleration from max-deacc to max-acc
         - a (continuous) direction with 3 values: 0) lane change to index -1, 1) no lane change,
                                                   2) lane change to index +1
        :return:
        """
        max_deacc = self.env_params.get_additional_param("max-deacc")
        max_acc = self.env_params.get_additional_param("max-acc")

        lb = [-abs(max_deacc), -1] * self.vehicles.num_rl_vehicles
        ub = [max_acc, 1] * self.vehicles.num_rl_vehicles
        return Box(np.array(lb), np.array(ub))

    @property
    def observation_space(self):
        """
        See parent class
        An observation consists of the velocity, lane index, and absolute position of each vehicle
        in the fleet
        """

        speed = Box(low=-np.inf, high=np.inf, shape=(self.vehicles.num_vehicles,))
        lane = Box(low=0, high=self.scenario.lanes-1, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, lane, absolute_pos))

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        # compute the system-level performance of vehicles from a velocity perspective
        reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        # punish excessive lane changes by reducing the reward by a set value every time an rl car changes lanes
        for veh_id in self.rl_ids:
            if self.vehicles.get_state(veh_id, "last_lc") == self.timer:
                reward -= 1

        return reward

    def get_state(self):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: an array of vehicle speed for each vehicle
        """
        return np.array([[self.vehicles.get_speed(veh_id) + normal(0, self.observation_vel_std),
                          self.vehicles.get_absolute_position(veh_id) + normal(0, self.observation_pos_std),
                          self.vehicles.get_lane(veh_id)] for veh_id in self.sorted_ids])

    def apply_rl_actions(self, actions):
        """
        Takes a tuple and applies a lane change or acceleration. if a lane change is applied,
        don't issue any commands for the duration of the lane change and return negative rewards
        for actions during that lane change. if a lane change isn't applied, and sufficient time
        has passed, issue an acceleration like normal
        :param actions: (acceleration, lc_value, direction)
        :return: array of resulting actions: 0 if successful + other actions are ok, -1 if unsucessful / bad actions.
        """
        # acceleration = actions[-1]
        # direction = np.array(actions[:-1]) - 1

        acceleration = actions[::2]
        direction = np.round(actions[1::2])

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = [self.timer <= self.lane_change_duration + self.vehicles.get_state(veh_id, 'last_lc')
                                 for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)


class LaneChangeOnlyEnvironment(SimpleLaneChangingAccelerationEnvironment):

    def __init__(self, env_params, sumo_params, scenario):

        super().__init__(env_params, sumo_params, scenario)

        # longitudinal (acceleration) controller used for rl cars
        self.rl_controller = dict()

        for veh_id in self.rl_ids:
            controller_params = env_params.get_additional_param("rl_acc_controller")
            self.rl_controller[veh_id] = controller_params[0](veh_id=veh_id, **controller_params[1])

    @property
    def action_space(self):
        """
        Actions are: a continuous direction for each rl vehicle
        """
        return Box(low=-1, high=1, shape=(self.vehicles.num_rl_vehicles,))

    def apply_rl_actions(self, actions):
        """
        see parent class
        - accelerations are derived using the IDM equation
        - lane-change commands are collected from rllab
        """
        direction = actions

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = [self.timer <= self.lane_change_duration + self.vehicles[veh_id]['last_lc']
                                 for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_lane_change(sorted_rl_ids, direction=direction)

        # collect the accelerations for the rl vehicles as specified by the human controller
        acceleration = []
        for veh_id in sorted_rl_ids:
            acceleration.append(self.rl_controller[veh_id].get_action(self))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)