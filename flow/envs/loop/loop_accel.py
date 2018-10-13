"""Environment for training the acceleration behavior of vehicles in a loop."""

from flow.envs.base_env import Env
from flow.core import rewards
from flow.core.params import InitialConfig, NetParams, SumoCarFollowingParams
from flow.controllers import IDMController, PISaturation

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
}


class AccelEnv(Env):
    """Fully observed acceleration environment.

    This environment used to train autonomous vehicles to improve traffic flows
    when acceleration actions are permitted by the rl agent.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function is the two-norm of the distance of the speed of the
        vehicles in the network from the "target_velocity" term. For a
        description of the reward, see: flow.core.rewards.desired_speed

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sumo_params, scenario)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(self.vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        obs_space = Box(
            low=0,
            high=1,
            shape=(self.vehicles.num_vehicles*2, ),
            dtype=np.float32)
        return obs_space

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.vehicles.get_rl_ids()
        ]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.vehicles.get_speed(self.vehicles.get_ids()))
        else:
            return rewards.desired_velocity(self, fail=kwargs["fail"])

    def get_state(self, **kwargs):
        """See class definition."""
        # speed normalizer
        max_speed = self.scenario.max_speed

        return np.array([[
            self.vehicles.get_speed(veh_id) / max_speed,
            self.get_x_by_id(veh_id) / self.scenario.length
        ] for veh_id in self.sorted_ids]).flatten()

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.vehicles.num_rl_vehicles > 0:
            for veh_id in self.vehicles.get_human_ids():
                self.vehicles.set_observed(veh_id)

class AccelCNNEnv(AccelEnv):
    @property
    def observation_space(self):
        """See class definition."""
        height = self.sights[0].shape[0]
        width = self.sights[0].shape[1]
        return Box(0., 1., [height, width, 5])

    def get_state(self, **kwargs):
        """See class definition."""
        sights_buffer = np.squeeze(np.array(self.sights_buffer))
        sights_buffer = np.moveaxis(sights_buffer, 0, -1)
        return sights_buffer / 255.

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.vehicles.num_rl_vehicles > 0:
            for veh_id in self.vehicles.get_human_ids():
                self.vehicles.set_observed(veh_id)

class AccelCNNIDMEnv(AccelCNNEnv):
    def __init__(self, env_params, sumo_params, scenario):
       AccelEnv.__init__(self, env_params, sumo_params, scenario)
       self.default_controller = [
           IDMController(vid, sumo_cf_params= SumoCarFollowingParams())
           for vid in self.vehicles.get_rl_ids()]

    def apply_acceleration(self, veh_ids, acc):
       for i, vid in enumerate(veh_ids):
           if acc[i] is not None:
               this_vel = self.vehicles.get_speed(vid)
               if "rl" in vid:
                   default_acc = self.default_controller[i].get_accel(self)
                   acc[i] += default_acc
               next_vel = max([this_vel + acc[i] * self.sim_step, 0])
               self.traci_connection.vehicle.slowDown(vid, next_vel, 1)

class AccelCNNPIEnv(AccelCNNEnv):
    def __init__(self, env_params, sumo_params, scenario):
       AccelEnv.__init__(self, env_params, sumo_params, scenario)
       self.default_controller = [
           PISaturation(vid, sumo_cf_params= SumoCarFollowingParams())
           for vid in self.vehicles.get_rl_ids()]

    def apply_acceleration(self, veh_ids, acc):
       for i, vid in enumerate(veh_ids):
           if acc[i] is not None:
               this_vel = self.vehicles.get_speed(vid)
               if "rl" in vid:
                   default_acc = self.default_controller[i].get_accel(self)
                   acc[i] += default_acc
               next_vel = max([this_vel + acc[i] * self.sim_step, 0])
               self.traci_connection.vehicle.slowDown(vid, next_vel, 1)
