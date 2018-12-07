from flow.envs.base_env import Env
from flow.envs.loop.wave_attenuation import WaveAttenuationEnv
from flow.envs.loop.loop_accel import AccelCNNIDMEnv
from flow.core.params import SumoCarFollowingParams
from flow.controllers import IDMController

import numpy as np
from flow.core import rewards
from flow.envs.base_env import Env
from flow.envs.multiagent_env import MultiEnv
from flow.core.params import InitialConfig, NetParams, SumoCarFollowingParams
from flow.controllers import IDMController

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np

import os
from os.path import expanduser
HOME = expanduser("~")
import time




class MinicityIDMEnv(Env):
    def __init__(self, env_params, sumo_params, scenario):
       super().__init__(env_params, sumo_params, scenario)
       self.default_controller = [
           IDMController(vid, sumo_cf_params=SumoCarFollowingParams())
           for vid in self.vehicles.get_rl_ids()]

    @property
    def action_space(self):
        num_actions = self.vehicles.num_rl_vehicles
        low=0
        high=float(10)
        return Box(low=low,high=high,shape=(num_actions,),dtype=np.float32)
        ##############################################################
        # specify dimensions and properties of the action space here #
        ##############################################################
        return  ### FILL IN ###

    @property
    def observation_space(self):

        return Box(
            low=0,
            high=float("inf"),
            shape=(3*self.vehicles.num_vehicles,))
        #############################################################
        # specify dimensions and properties of the state space here #
        #############################################################


    def get_state(self, **kwargs):
        ids = self.vehicles.get_ids()

        # we use the get_absolute_position method to get the positions of all vehicles
        headway = np.divide([self.vehicles.get_headway(veh_id) for veh_id in ids],10)

        # we use the get_absolute_position method to get the positions of all vehicles
        vel = np.divide([self.vehicles.get_speed(veh_id) for veh_id in ids],15)

        speed_lead=np.divide([self.vehicles.get_speed(self.vehicles.get_leader(veh_id)) for veh_id in ids],15)
        # the speeds and headway are concatenated to produce the state


        
        print(ids)

        return np.concatenate((headway, vel,speed_lead))
        ####################################
        # specify desired state space here #
        ####################################


    def _apply_rl_actions(self, rl_actions):
        #####################################
        # specify desired action space here #
        #####################################
        ### FILL IN ###

        # the names of all autonomous (RL) vehicles in the network
        rl_ids = self.vehicles.get_rl_ids()

        # use the base environment method to convert actions into accelerations for the rl vehicles
        self.apply_acceleration(rl_ids, rl_actions)
        return None

    def apply_acceleration(self, veh_ids, acc):
       for i, vid in enumerate(veh_ids):
           if acc[i] is not None:
               this_vel = self.vehicles.get_speed(vid)
               if "rl" in vid:
                   low=-np.abs(self.env_params.additional_params["max_decel"])
                   high=self.env_params.additional_params["max_accel"]
                   alpha = self.env_params.additional_params["augmentation"]
                   default_acc = self.default_controller[i].get_accel(self)
                   #acc[i] = alpha*acc[i] +(1.0 - alpha)*np.clip(default_acc, low, high)
                   #import pdb; pdb.set_trace()
                   acc[i] = (float(alpha)*np.array([acc[i]]).astype(np.float32) +(1.0 - float(alpha)) * np.array([np.clip(default_acc, low, high)]).astype(np.float32))
               next_vel = max([this_vel + acc[i] * self.sim_step, 0])
               self.traci_connection.vehicle.slowDown(vid, next_vel, 1)


    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        max_speed = 10
        speed = self.vehicles.get_speed(self.vehicles.get_ids())
        return (0.8*np.mean(speed) - 0.2*np.std(speed))/max_speed




class MinicityCNNIDMEnv(AccelCNNIDMEnv):
    def __init__(self, env_params, sumo_params, scenario):
       super().__init__(env_params, sumo_params, scenario)
       self.default_controller = [
           IDMController(vid, sumo_cf_params=SumoCarFollowingParams())
           for vid in self.vehicles.get_rl_ids()]

    def apply_acceleration(self, veh_ids, acc):
       for i, vid in enumerate(veh_ids):
           if acc[i] is not None:
               this_vel = self.vehicles.get_speed(vid)
               if "rl" in vid:
                   low=-np.abs(self.env_params.additional_params["max_decel"])
                   high=self.env_params.additional_params["max_accel"]
                   alpha = self.env_params.additional_params["augmentation"]
                   default_acc = self.default_controller[i].get_accel(self)
                   #acc[i] = alpha*acc[i] +(1.0 - alpha)*np.clip(default_acc, low, high)
                   acc[i] = (float(alpha)*np.array([acc[i]]).astype(np.float32) +(1.0 - float(alpha)) * np.array([np.clip(default_acc, low, high)]).astype(np.float32))
               next_vel = max([this_vel + acc[i] * self.sim_step, 0])
               self.traci_connection.vehicle.slowDown(vid, next_vel, 1)
