"""Example of modified minicity network with human-driven vehicles."""

import ray
import ray.rllib.agents.es as es
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog, Model
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv
from flow.envs.minicity import MinicityIDMEnv
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import MinicityTrainingRouter_9
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
import json
from flow.controllers.routing_controllers import MinicityTrainingRouter_4

import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 10,
}

augmentation = sys.argv[1]
ADDITIONAL_ENV_PARAMS['augmentation']=augmentation


# time horizon of a single rollout
HORIZON = 1
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 2


additional_net_params = ADDITIONAL_NET_PARAMS.copy()
"""
Perform a simulation of vehicles on modified minicity of University of
Delaware.

Parameters
----------
render: bool, optional
    specifies whether to use sumo's gui during execution

Returns
-------
exp: flow.core.SumoExperiment type
    A non-rl experiment demonstrating the performance of human-driven
    vehicles on the minicity scenario.
"""
#sumo_params = SumoParams(render='drgb',
#save_render=False,
#sight_radius=20,
#pxpm=3,
#show_radius=True)
#
#if render is not None:
#    sumo_params.render = render
#
#if save_render is not None:
#    sumo_params.save_render = save_render
#
#if sight_radius is not None:
#    sumo_params.sight_radius = sight_radius
#
#if pxpm is not None:
#    sumo_params.pxpm = pxpm
#
#if show_radius is not None:
#    sumo_params.show_radius = show_radius
#
# sumo_params.sim_step = 0.2

vehicles = Vehicles()

edge_starts = ['e_80', 'e_83', 'e_82', 'e_79', 'e_47', 'e_49', 'e_55',
               'e_56', 'e_89', 'e_45', 'e_43', 'e_41', 'e_50', 'e_60',
               'e_69', 'e_73', 'e_75', 'e_86', 'e_59', 'e_48', 'e_81',
               'e_84', 'e_85', 'e_90', 'e_62', 'e_57', 'e_46', 'e_76',
               'e_76', 'e_74', 'e_70', 'e_61', 'e_54', 'e_40', 'e_42',
               'e_44']
# bottom-left
edge_starts += ['e_25', 'e_30', 'e_31', 'e_32', 'e_21', 'e_8_u', 'e_9',
                'e_10', 'e_11', 'e_87', 'e_39', 'e_37', 'e_29_u', 'e_92',
                'e_7', 'e_8_b', 'e_10']
# upper left
edge_starts += ['e_12', 'e_18', 'e_19', 'e_24', 'e_45', 'e_43',
                'e_41', 'e_88', 'e_26', 'e_34', 'e_23', 'e_5', 'e_4',
                'e_3', 'e_25', 'e_87', 'e_40', 'e_42', 'e_44', 'e_15',
                'e_16', 'e_20', 'e_47', 'e_46']
# bottom right corner
edge_starts += ['e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63',
                'e_94', 'e_52', 'e_38']
# bottom half outer loop
edge_starts += ['e_67', 'e_71', 'e_70', 'e_61', 'e_54', 'e_88', 'e_26',
                'e_2', 'e_1', 'e_7', 'e_17', 'e_28_b', 'e_36', 'e_93',
                'e_53', 'e_64']
# bottom right inner loop
edge_starts += ['e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63',
                'e_94', 'e_52', 'e_38']



edge_starts = list(set(edge_starts))


vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(MinicityTrainingRouter_4, {}),
    speed_mode='no_collide',
    lane_change_mode='strategic',
    num_vehicles=1)
"""
vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {}),
    routing_controller=(MinicityTrainingRouter_4, {}),
    speed_mode='no_collide',
    lane_change_mode='strategic',
    num_vehicles=30)
"""
flow_params = dict(
    # name of the experiment
    exp_tag="mincity_v0_%s" % augmentation,

    # name of the flow environment the experiment is running on
    env_name="MinicityIDMEnv",

    # name of the scenario class the experiment is running on
    scenario="MiniCityScenario",



    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(render=False,sim_step=1,restart_instance=True),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(additional_params=ADDITIONAL_ENV_PARAMS,
        horizon=HORIZON
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False, additional_params=additional_net_params
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial_config = InitialConfig(
        spacing='random',
        edges_distribution=edge_starts,
        min_gap=2),
)

# initial_config = InitialConfig(
#     spacing="random",
#     min_gap=5
# )
#scenario = MiniCityScenario(
#    name='minicity',
#    vehicles=vehicles,
#    initial_config=initial_config,
#    net_params=net_params)
#
#env = AccelEnv(env_params, sumo_params, scenario)
#
#return SumoExperiment(env, scenario)


if __name__ == "__main__":
    ray.init(num_cpus=N_CPUS, num_gpus=0, redirect_output=False)

    config = es.DEFAULT_CONFIG.copy()
    config["episodes_per_batch"] = N_ROLLOUTS
    config["num_workers"] = N_ROLLOUTS
    config["eval_prob"] = 0.05
    config["noise_stdev"] = 0.01

    config["stepsize"] = 0.01
    config["observation_filter"] = "NoFilter"
    config['horizon']=HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": "ES",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 10,
            "max_failures": 999,
            "stop": {
                "training_iteration": 100,
            },
            "num_samples": 6,
        },
    })
