"""Environment for training the acceleration behavior of vehicles in a loop."""

from flow.envs.base_env import Env
from flow.core import rewards
from flow.core.params import InitialConfig, NetParams, SumoCarFollowingParams
from flow.controllers import IDMController

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np

import os
from os.path import expanduser
HOME = expanduser("~")
import time

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 11.176,
}


class SoftIntersectionEnv(Env):
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

        # setup observation cache
        self.occupancy_table = np.zeros((16, 5))
        self.vehicle_index = {}

    # ACTION GOES HERE
    @property
    def action_space(self):
        return Box(
            low=0,
            high=81,
            shape=(3,),
            dtype=np.float32)

    def set_action(self, action):
        agent_idx = action[0]
        if agent_idx < 80:
            # acting on vehicles
            max_accel, min_decel = 1.00, -3.00
            lower, upper = 0.5, 1.5
            mu, sigma = action[1], action[2]
            speed_multiplier = np.clip(
                np.random.normal(loc=mu, scale=sigma), lower, upper
            )
            if agent_idx in self.vehicle_index.keys():
                veh_list = self.vehicle_index[agent_idx]
                for veh_id in veh_list:
                    veh_speed = self.traci_connection.vehicle.getSpeed(veh_id)
                    max_speed = veh_speed + max_accel
                    min_speed = max(veh_speed + min_decel, 0)
                    veh_speed = veh_speed * speed_multiplier
                    veh_speed = np.clip(veh_speed, min_speed, max_speed)
                    self.traci_connection.vehicle.slowDown(
                        veh_id, veh_speed, 1000
                    )

        elif agent_idx == 80:
            # acting on traffic lights
            lower, upper = 1.0, self.tls_phase_count - 1
            mu, sigma = action[1], action[2]
            tls_phase_increment = np.round(np.clip(
                np.random.normal(loc=mu, scale=sigma), lower, upper
            ))
            self.tls_phase = \
                self.traci_connection.trafficlight.getPhase(self.tls_id)
            self.tls_phase += tls_phase_increment
            self.tls_phase %= self.tls_phase_count
            self.traci_connection.trafficlight.setPhase(\
                self.tls_id, tls_phase)
        elif agent_idx == 81:
            # no ops
            pass
        else:
            raise ValueError('Agent index exceeds 81.')

    # OBSERVATION GOES HERE
    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0.,
            high=np.inf,
            shape=(81,),
            dtype=np.float32)

    def get_observation(self, **kwargs):
        observation = self.occupancy_table.flatten().tolist()
        tls_phase = self.tls_phase
        observation = observation + [tls_phase]
        return np.asarray(observation)

    # REWARD FUNCTION GOES HERE
    def get_reward(self, **kwargs):
        # safety reward (WARNING: sum_collisions is not working yet.)
        # _sum_collisions = self.sum_collisions * 1  # TODO: normalize
        # print("_sum_collisions =", _sum_collisions)
        # _min_headway = self.min_headway * 1  # TODO: normalize
        # print("_min_headway =", _min_headway)
        # _safety = 0.8 * _sum_collisions + 0.2 * _min_headway
        # print("_safety =", _safety)

        # performance reward
        _avg_speed = self.avg_speed * 1  # TODO: normalize
        _std_speed = self.std_speed * 1  # TODO: normalize
        _performance = 0.8 * _avg_speed + 0.2 * _std_speed

        # consumption reward
        _avg_fuel = self.avg_fuel * -10  # TODO: normalize
        _avg_co2 = self.avg_co2 * -5e-4  # TODO: normalize
        _cost = 0.5 * _avg_fuel + 0.5 * _avg_co2

        # total reward
        #reward = 0.5 * _safety + 0.4 * _performance + 0.1 * _cost
        reward = 0.5 * _performance + 0.5 * _cost
        if np.isnan(reward):
            reward = 0

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
        self.occupancy_table = np.zeros((16, 5))
        self.vehicle_index = {}
        for veh_id in self.vehicles.get_ids():
            edge_id = self.traci_connection.vehicle.getRoadID(veh_id)
            if 'in' in edge_id or 'out' in edge_id:
                continue
            veh_type = self.traci_connection.vehicle.getTypeID(veh_id)
            route = self.traci_connection.vehicle.getRoute(veh_id)
            edge_idx = int(route[0][2]) // 2
            row_idx = edge_idx * 4
            if veh_type == 'autonomous':
                route_idx = self.route_idx_table[(route[1], route[2])]
            else:
                route_idx = 3
            row_idx += route_idx
            col_idx = self.vehicles.get_position(veh_id) // 20
            col_idx = int(min(col_idx, 4))
            try:
                self.occupancy_table[row_idx, col_idx] += 1
                agent_idx = int(row_idx*5 + col_idx)
                if agent_idx in self.vehicle_index.keys():
                    self.vehicle_index[agent_idx] += [veh_id]
                else:
                    self.vehicle_index[agent_idx] = [veh_id]
            except IndexError:
                raise IndexError(veh_id, veh_type, route,
                      self.vehicles.get_position(veh_id))

        # Update key attributes
        self.sum_collisions = \
            self.traci_connection.simulation.getCollidingVehiclesNumber()
        self.min_headway = np.inf
        speeds = []
        fuels = []
        co2s = []
        for veh_id in self.vehicles.get_ids():
            headway = self.vehicles.get_headway(veh_id)
            if headway < self.min_headway:
                self.min_headway = headway
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

    # DO NOT WORRY ABOUT ANYTHING BELOW THIS LINE >◡<
    def _apply_rl_actions(self, rl_actions):
        self.set_action(rl_actions)

    def get_state(self, **kwargs):
        return self.get_observation(**kwargs)

    def compute_reward(self, actions, **kwargs):
        return self.get_reward(**kwargs)

class HardIntersectionEnv(Env):
    def __init__(self, env_params, sumo_params, scenario):
        print("Starting HardIntersectionEnv...")
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sumo_params, scenario)

        # setup traffic lights
        self.tls_id = self.traci_connection.trafficlight.getIDList()[0]
        self.tls_state =\
            self.traci_connection.trafficlight.\
            getRedYellowGreenState(self.tls_id)
        self.tls_definition =\
            self.traci_connection.trafficlight.\
            getCompleteRedYellowGreenDefinition(self.tls_id)
        self.tls_phase = 0
        self.tls_phase_count = 0
        for logic in self.tls_definition:
            for phase in logic._phases:
                self.tls_phase_count += 1
        self.tls_phase_increment = 0

        # setup speed broadcasters
        self.sbc_locations = [
            "e_1_zone1+_0", "e_1_zone1+_1",  # east bound
            "e_1_zone2+_0", "e_1_zone2+_1",  # east bound
            "e_1_zone3+_0", "e_1_zone3+_1",  # east bound
            "e_1_zone4+_0", "e_1_zone4+_1",  # east bound

            "e_2_zone1+_0", "e_2_zone1+_1",  # south bound
            "e_2_zone2+_0", "e_2_zone2+_1",  # south bound
            "e_2_zone3+_0", "e_2_zone3+_1",  # south bound
            "e_2_zone4+_0", "e_2_zone4+_1",  # south bound

            "e_3_zone1+_0", "e_3_zone1+_1",  # west bound
            "e_3_zone2+_0", "e_3_zone2+_1",  # west bound
            "e_3_zone3+_0", "e_3_zone3+_1",  # west bound
            "e_3_zone4+_0", "e_3_zone4+_1",  # west bound

            "e_4_zone1+_0", "e_4_zone1+_1",  # north bound
            "e_4_zone2+_0", "e_4_zone2+_1",  # north bound
            "e_4_zone3+_0", "e_4_zone3+_1",  # north bound
            "e_4_zone4+_0", "e_4_zone4+_1",  # north bound
        ]
        # default speed reference to 11.176 m/s
        self.sbc_command = {
            loc: self.traci_connection.lane.getMaxSpeed(loc)
            for loc in self.sbc_locations
        }

        # setup inflow outflow logger
        self.inflow_locations = [
            "e_1_zone1+_0", "e_1_zone1+_1",  # east bound
            "e_1_zone2+_0", "e_1_zone2+_1",  # east bound
            "e_1_zone3+_0", "e_1_zone3+_1",  # east bound
            "e_1_zone4+_0", "e_1_zone4+_1",  # east bound

            "e_2_zone1+_0", "e_2_zone1+_1",  # south bound
            "e_2_zone2+_0", "e_2_zone2+_1",  # south bound
            "e_2_zone3+_0", "e_2_zone3+_1",  # south bound
            "e_2_zone4+_0", "e_2_zone4+_1",  # south bound

            "e_3_zone1+_0", "e_3_zone1+_1",  # west bound
            "e_3_zone2+_0", "e_3_zone2+_1",  # west bound
            "e_3_zone3+_0", "e_3_zone3+_1",  # west bound
            "e_3_zone4+_0", "e_3_zone4+_1",  # west bound

            "e_4_zone1+_0", "e_4_zone1+_1",  # north bound
            "e_4_zone2+_0", "e_4_zone2+_1",  # north bound
            "e_4_zone3+_0", "e_4_zone3+_1",  # north bound
            "e_4_zone4+_0", "e_4_zone4+_1",  # north bound
        ]
        self.inflow_accelerations = {loc: 0 for loc in self.inflow_locations}
        self.inflow_speeds = {loc: 0 for loc in self.inflow_locations}
        self.inflow_densities = { loc: 0 for loc in self.inflow_locations}
        self.inflow_fuels = {loc: 0 for loc in self.inflow_locations}
        self.inflow_co2s = {loc: 0 for loc in self.inflow_locations}
        self.outflow_locations = [
            "e_1_zone1-_0", "e_1_zone1-_1",  # east bound
            "e_1_zone2-_0", "e_1_zone2-_1",  # east bound
            "e_1_zone3-_0", "e_1_zone3-_1",  # east bound
            "e_1_zone4-_0", "e_1_zone4-_1",  # east bound

            "e_2_zone1-_0", "e_2_zone1-_1",  # south bound
            "e_2_zone2-_0", "e_2_zone2-_1",  # south bound
            "e_2_zone3-_0", "e_2_zone3-_1",  # south bound
            "e_2_zone4-_0", "e_2_zone4-_1",  # south bound

            "e_3_zone1-_0", "e_3_zone1-_1",  # west bound
            "e_3_zone2-_0", "e_3_zone2-_1",  # west bound
            "e_3_zone3-_0", "e_3_zone3-_1",  # west bound
            "e_3_zone4-_0", "e_3_zone4-_1",  # west bound

            "e_4_zone1-_0", "e_4_zone1-_1",  # north bound
            "e_4_zone2-_0", "e_4_zone2-_1",  # north bound
            "e_4_zone3-_0", "e_4_zone3-_1",  # north bound
            "e_4_zone4-_0", "e_4_zone4-_1",  # north bound
        ]
        self.outflow_accelerations = {loc: 0 for loc in self.outflow_locations}
        self.outflow_speeds = {loc: 0 for loc in self.outflow_locations}
        self.outflow_densities = {loc: 0 for loc in self.outflow_locations}
        self.outflow_fuels = {loc: 0 for loc in self.outflow_locations}
        self.outflow_co2s = {loc: 0 for loc in self.outflow_locations}

        # setup reward-related variables
        self.alpha = env_params.additional_params["alpha"]
        self.rewards = 0

    # ACTION GOES HERE
    @property
    def action_space(self):
        return Box(
            low=0,
            high=max(self.scenario.max_speed, self.tls_phase_count),
            shape=(33,),
            dtype=np.float32)

    def set_action(self, action):
        self.sbc_command = {
            loc: np.clip(action[idx], 0, np.inf)
            for idx, loc in enumerate(self.sbc_locations)
        }
        self.tls_phase_increment = np.clip(
            int(action[-1]), 0, self.tls_phase_count)
        self._set_command(self.sbc_command)
        self.tls_phase += self.tls_phase_increment
        self.tls_phase %= self.tls_phase_count
        self._set_phase(self.tls_phase)

    # OBSERVATION GOES HERE
    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0.,
            high=np.inf,
            shape=(193,),
            dtype=np.float32)

    def get_observation(self, **kwargs):
        inflow_accelerations = [
            self.inflow_accelerations[loc]
            for loc in self.inflow_locations
        ]
        inflow_speeds = [
            self.inflow_speeds[loc]
            for loc in self.inflow_locations
        ]
        inflow_densities = [
            self.inflow_densities[loc]
            for loc in self.inflow_locations
        ]
        outflow_accelerations = [
            self.outflow_accelerations[loc]
            for loc in self.outflow_locations
        ]
        outflow_speeds = [
            self.outflow_speeds[loc]
            for loc in self.outflow_locations
        ]
        outflow_densities = [
            self.outflow_densities[loc]
            for loc in self.outflow_locations
        ]
        tls_phase = self.tls_phase
        observation = np.asarray(
            inflow_accelerations + inflow_speeds + inflow_densities +
            outflow_accelerations + outflow_speeds + outflow_densities +
            [tls_phase]
        )
        return observation

    # REWARD FUNCTION GOES HERE
    def get_reward(self, **kwargs):
        speeds = list(self.inflow_speeds.values())
        speeds += list(self.outflow_speeds.values())
        densities = list(self.inflow_densities.values())
        densities += list(self.outflow_densities.values())
        performance = 0.4*np.mean(speeds) + 0.1*-np.std(speeds) + \
                      0.4*-np.mean(densities) + 0.1*-np.std(densities)
        fuels = list(self.inflow_fuels.values())
        fuels += list(self.outflow_fuels.values())
        co2s = list(self.inflow_co2s.values())
        co2s += list(self.outflow_co2s.values())
        consumption = 0.5*-np.mean(fuels) + 0.5*-np.mean(co2s)/1e2
        return self.alpha * performance + (1 - self.alpha) * consumption

    # UTILITY FUNCTION GOES HERE
    def additional_command(self):
        # update inflow statistics
        inflow_stats = []
        for idx, loc in enumerate(self.inflow_locations):
            flow_stats = self.get_flow_stats(loc)
            inflow_stats.append(flow_stats)
            acceleration, speed, _, _, density, fuel, co2 = flow_stats
            self.inflow_accelerations[loc] = acceleration
            self.inflow_speeds[loc] = speed
            self.inflow_densities[loc] = density
            self.inflow_fuels[loc] = fuel
            self.inflow_co2s[loc] = co2

        # update outflow statistics
        outflow_stats = []
        for idx, loc in enumerate(self.outflow_locations):
            flow_stats = self.get_flow_stats(loc)
            outflow_stats.append(flow_stats)
            acceleration, speed, _, _, density, fuel, co2 = flow_stats
            self.outflow_accelerations[loc] = acceleration
            self.outflow_speeds[loc] = speed
            self.outflow_densities[loc] = density
            self.outflow_fuels[loc] = fuel
            self.outflow_co2s[loc] = co2

        # update traffic lights state
        self.tls_state =\
            self.traci_connection.trafficlight.\
            getRedYellowGreenState(self.tls_id)

        # disable skip to test traci tls and sbc setter methods
        self.test_sbc(skip=True)
        self.test_tls(skip=True)
        self.test_ioflow(inflow_stats, outflow_stats, skip=True)
        self.test_reward(skip=True)

    def test_sbc(self, skip=True):
        if self.time_counter > 50 and not skip:
            print("Broadcasting command...")
            self.sbc_command = {
                loc: 1
                for loc in self.sbc_locations
            }
            self._set_command(self.sbc_command)

    def test_tls(self, skip=True):
        if self.time_counter % 10 == 0 and not skip:
            print("Switching phase...")
            self.tls_phase = np.random.randint(0, self.tls_phase_count-1)
            print("New phase:", self.tls_phase)
            self._set_phase(self.tls_phase)

    def test_ioflow(self, inflow_stats, outflow_stats, skip=False):
        if not skip:
            print("inflow:", inflow_stats)
            print("acceleration:", self.inflow_accelerations)
            print("speed:", self.inflow_speeds)
            print("density:", self.inflow_densities)
            print("fuel:", self.inflow_fuels)
            print("co2:", self.inflow_co2s)

            print("outflow:", outflow_stats)
            print("acceleration:", self.outflow_accelerations)
            print("speed:", self.outflow_speeds)
            print("density:", self.outflow_densities)
            print("fuel:", self.outflow_fuels)
            print("co2:", self.outflow_co2s)

    def test_reward(self, skip=True):
        if not skip:
            _reward = self.get_reward()
            self.rewards += _reward

    def get_flow_stats(self, loc):
        speed = self.traci_connection.lane.getLastStepMeanSpeed(loc)
        try:
            acceleration = (speed - self.inflow_speeds[loc])/self.sim_step
        except KeyError:
            acceleration = (speed - self.outflow_speeds[loc])/self.sim_step
        count = self.traci_connection.lane.getLastStepVehicleNumber(loc)
        length = self.traci_connection.lane.getLength(loc)
        lane_vehicles = self.traci_connection.lane.getLastStepVehicleIDs(loc)
        density = count / length
        fuel = self.traci_connection.lane.getFuelConsumption(loc) / length
        co2 = self.traci_connection.lane.getCO2Emission(loc) / length
        if count == 0:
            speed = 0
        return [
            acceleration, speed, count, length, density, fuel, co2
        ]

    def _set_phase(self, tls_phase):
        self.traci_connection.trafficlight.setPhase(\
            self.tls_id, tls_phase)

    def _set_command(self, sbc_command):
        for sbc, reference in sbc_command.items():
            sbc_clients = self.traci_connection.lane.getLastStepVehicleIDs(sbc)
            for veh_id in sbc_clients:
                self.traci_connection.vehicle.setSpeed(veh_id, reference)

    # DO NOT WORRY ABOUT ANYTHING BELOW THIS LINE >◡<
    def _apply_rl_actions(self, rl_actions):
        self.set_action(rl_actions)

    def get_state(self, **kwargs):
        return self.get_observation(**kwargs)

    def compute_reward(self, actions, **kwargs):
        return self.get_reward(**kwargs)
