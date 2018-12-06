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
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.envs.minicity import MinicityCNNIDMEnv
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


augmentation = sys.argv[1]
ADDITIONAL_ENV_PARAMS['augmentation']=augmentation

render='drgb',
save_render=False,
sight_radius=20,
pxpm=3,
show_radius=True

# time horizon of a single rollout
HORIZON = 100
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

edge_starts = ['e_79']


edge_starts = list(set(edge_starts))


vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(MinicityTrainingRouter_4, {}),
    speed_mode='no_collide',
    lane_change_mode='strategic',
    num_vehicles=1)


vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {}),
    routing_controller=(MinicityTrainingRouter_4, {}),
    speed_mode='no_collide',
    lane_change_mode='strategic',
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="mincity_v0_%s" % augmentation,

    # name of the flow environment the experiment is running on
    env_name="MinicityIDMEnv",

    # name of the scenario class the experiment is running on
    scenario="MiniCityScenario",



    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(render=True,
    save_render=False,
    sight_radius=20,
    pxpm=3,
    show_radius=True),

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
