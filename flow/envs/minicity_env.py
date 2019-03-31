from flow.envs.base_env import Env
from gym.spaces.box import Box
import numpy as np

ADDITIONAL_ENV_PARAMS = {}


class MinicityEnv(Env):
    def __init__(self, env_params, sumo_params, scenario):
        print("Starting MinicityEnv...")
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sumo_params, scenario)
        self.lane_id_list = []
        for lane_id in self.traci_connection.lane.getIDList():
            self.lane_id_list.append(lane_id)
        self.num_lanes = len(self.lane_id_list)
        self.lane_dynamics = np.zeros((self.num_lanes,))

    # ACTION GOES HERE
    @property
    def action_space(self):
        action = Box(
            low=11.18*0.1,
            high=11.18*2,
            shape=(self.num_lanes,),
            dtype=np.float32,
        )
        return action

    def set_action(self, action):
        for idx, lane_id in enumerate(self.lane_id_list):
            self.traci_connection.lane.setMaxSpeed(lane_id, action[idx])

    # OBSERVATION GOES HERE
    @property
    def observation_space(self):
        """See class definition."""
        observation = Box(
            low=11.18*0.1,
            high=11.18*2,
            shape=(self.num_lanes,),
            dtype=np.float32,
        )
        return observation

    def get_observation(self, **kwargs):
        lane_dynamics = []
        for lane_id in self.lane_id_list:
            _lane_speed = \
                self.traci_connection.lane.getLastStepMeanSpeed(lane_id)
            lane_dynamics.append(_lane_speed)
        self.lane_dynamics = np.asarray(lane_dynamics)
        return self.lane_dynamics

    # REWARD FUNCTION GOES HERE
    def get_reward(self, **kwargs):
        self.reward = \
            0.8 * self.lane_dynamics.mean() - 0.2 * self.lane_dynamics.std()
        return self.reward

    # UTILITY FUNCTION GOES HERE
    def additional_command(self):
        pass

    # DO NOT WORRY ABOUT ANYTHING BELOW THIS LINE >◡<
    def _apply_rl_actions(self, rl_actions):
        self.set_action(rl_actions)

    def get_state(self, **kwargs):
        return self.get_observation(**kwargs)

    def compute_reward(self, actions, **kwargs):
        return self.get_reward(**kwargs)
