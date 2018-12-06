from flow.envs.loop.loop_accel import AccelCNNIDMEnv,AccelEnv
from flow.core.params import SumoCarFollowingParams
from flow.controllers import IDMController

import numpy as np

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

class MinicityIDMEnv(AccelEnv):
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
                   #import pdb; pdb.set_trace()
                   acc[i] = (float(alpha)*np.array([acc[i]]).astype(np.float32) +(1.0 - float(alpha)) * np.array([np.clip(default_acc, low, high)]).astype(np.float32))
               next_vel = max([this_vel + acc[i] * self.sim_step, 0])
               self.traci_connection.vehicle.slowDown(vid, next_vel, 1)
