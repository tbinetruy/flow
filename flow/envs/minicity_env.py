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
        self.lane_id_dict = {}
        self.lane_idx_dict = {}
        lane_idx = 0
        for lane_id in self.traci_connection.lane.getIDList():
            if 'e' in lane_id:
                self.lane_id_list.append(lane_id)
                self.lane_id_dict[lane_id] = lane_idx
                self.lane_idx_dict[lane_idx] = lane_id
                lane_idx += 1
        self.num_lanes = len(self.lane_id_list)
        self.lane_dynamics = np.zeros((self.num_lanes,))
        self.adjacency_matrix = np.zeros((self.num_lanes, self.num_lanes))
        for lane_id in self.lane_id_list:
            current_idx = self.lane_id_dict[lane_id]
            for link in self.traci_connection.lane.getLinks(lane_id):
                next_lane_id = link[0]
                next_idx = self.lane_id_dict[next_lane_id]
                self.adjacency_matrix[current_idx, next_idx] = 1
        # Split equally among all directions
        self.transition_matrix = self.adjacency_matrix.copy()
        for row in range(self.num_lanes):
            self.transition_matrix[row, :] /= \
                np.sum(self.adjacency_matrix[row, :])

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

class MinicityMatrixEnv(MinicityEnv):
    def __init__(self, env_params, sumo_params, scenario):
        print("Starting MinicityMatrixEnv...")
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        super().__init__(env_params, sumo_params, scenario)

        self.lane_id_list = []
        self.lane_id_dict = {}
        self.lane_idx_dict = {}
        lane_idx = 0
        for lane_id in self.traci_connection.lane.getIDList():
            if 'e' in lane_id:
                self.lane_id_list.append(lane_id)
                self.lane_id_dict[lane_id] = lane_idx
                self.lane_idx_dict[lane_idx] = lane_id
                lane_idx += 1
        self.num_lanes = len(self.lane_id_list)
        self.lane_dynamics = np.zeros((self.num_lanes,))
        self.adjacency_matrix = np.zeros((self.num_lanes, self.num_lanes))
        for lane_id in self.lane_id_list:
            current_idx = self.lane_id_dict[lane_id]
            for link in self.traci_connection.lane.getLinks(lane_id):
                next_lane_id = link[0]
                next_idx = self.lane_id_dict[next_lane_id]
                self.adjacency_matrix[current_idx, next_idx] = 1
        self.nonzero_junctions = np.nonzero(self.adjacency_matrix)

        # Split equally among all directions
        self.transition_matrix = self.adjacency_matrix.copy()
        for row in range(self.num_lanes):
            self.transition_matrix[row, :] /= \
                np.sum(self.adjacency_matrix[row, :])

        debug_mode = False
        if debug_mode:
            flat_trans_prob = self.transition_matrix.flatten()
            pos_flat_trans_prob = flat_trans_prob[flat_trans_prob!=0]
            print(pos_flat_trans_prob)
            print(len(pos_flat_trans_prob))

            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            im = ax.imshow(
                self.transition_matrix, cmap='plasma', vmin=0.0, vmax=1.0)
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_xticks(np.arange(self.num_lanes))
            ax.set_xticklabels(self.lane_id_list)
            ax.set_yticks(np.arange(self.num_lanes))
            ax.set_yticklabels(self.lane_id_list)
            plt.show()

    # ACTION GOES HERE
    @property
    def action_space(self):
        action = Box(
            low=0.1,
            high=1.0,
            shape=(np.count_nonzero(self.adjacency_matrix),),
            dtype=np.float32,
        )
        return action

    def set_action(self, action):
        if self.time_counter % 100 == 0:
            self.transition_matrix = np.zeros_like(self.adjacency_matrix)
            self.transition_matrix[self.nonzero_junctions] = action
            for row in range(self.num_lanes):
                self.transition_matrix[row, :] /= \
                    np.sum(self.transition_matrix[row, :])

    # OBSERVATION GOES HERE
    @property
    def observation_space(self):
        """See class definition."""
        observation = Box(
            low=0,
            high=self.scenario.max_speed,
            shape=(self.num_lanes,),
            dtype=np.float32,
        )
        return observation

    def get_observation(self, **kwargs):
        if self.time_counter % 100 == 0:
            lane_dynamics = []
            for lane_id in self.lane_id_list:
                _lane_speed = \
                    self.traci_connection.lane.getLastStepMeanSpeed(lane_id)
                lane_dynamics.append(_lane_speed)
            self.lane_dynamics = np.asarray(lane_dynamics)
        return self.lane_dynamics

    # REWARD FUNCTION GOES HERE
    def get_reward(self, **kwargs):
        if self.time_counter % 100 == 0:
            self.reward = \
                0.75 * self.lane_dynamics.mean() - 0.25 * self.lane_dynamics.std()
            return self.reward
        else:
            return 0

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
