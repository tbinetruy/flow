"""Intersection (S)ingle (A)gent (R)einforcement (L)earning."""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers import IDMController, ContinuousRouter,\
    SumoCarFollowingController, SumoLaneChangeController
from flow.controllers.routing_controllers import IntersectionRouter
from flow.envs.intersection_env import SoftIntersectionEnv, \
    ADDITIONAL_ENV_PARAMS
from flow.scenarios.intersection import \
    SoftIntersectionScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import IntersectionRandomRouter
from flow.core.params import InFlows
import numpy as np
seed=204
np.random.seed(seed)

# time horizon of a single rollout
HORIZON = 5000
# number of rollouts per training iteration
N_ROLLOUTS = 18
# number of parallel workers
N_CPUS = 6

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()

# We place 40 autonomous vehicles in the network
vehicles = Vehicles()

# Add mixed-autonomy traffic
insertion_prob = 0.1
autonomy_percent = 0.5
prob_table = {
    'manned': (1 - autonomy_percent)*insertion_prob,
    'autonomous': autonomy_percent*insertion_prob,
}
inflow = InFlows()
for type in ['manned', 'autonomous']:
    vehicles.add(
        veh_id=type,
        speed_mode=0b11111,
        lane_change_mode=0b011001010101,
        acceleration_controller=(SumoCarFollowingController, {}),
        lane_change_controller=(SumoLaneChangeController, {}),
        routing_controller=(IntersectionRandomRouter, {}),
        num_vehicles=0,
    )
    inflow.add(
        veh_type=type,
        edge='e_1_in',
        probability=prob_table[type],
        departSpeed=8,
        departLane='random'
    )
    inflow.add(
        veh_type=type,
        edge='e_3_in',
        probability=prob_table[type],
        departSpeed=8,
        departLane='random'
    )
    inflow.add(
        veh_type=type,
        edge='e_5_in',
        probability=prob_table[type],
        departSpeed=8,
        departLane='random'
    )
    inflow.add(
        veh_type=type,
        edge='e_7_in',
        probability=prob_table[type],
        departSpeed=8,
        departLane='random'
    )

flow_params = dict(
    # name of the experiment
    exp_tag='intersection-sarl-soft',

    # name of the flow environment the experiment is running on
    env_name='SoftIntersectionEnv',

    # name of the scenario class the experiment is running on
    scenario='SoftIntersectionScenario',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.1,
        render=False,
        seed=seed,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=1,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        junction_type='traffic_light',
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='uniform',
        edges_distribution=['e_1'],
        min_gap=5,
    ),
)


def setup_exps():

    alg_run = 'ES' # TODO: Make this an argument
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['horizon'] = HORIZON
    config['episodes_per_batch'] = N_ROLLOUTS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == '__main__':
    alg_run, gym_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1, redirect_output=False)
    trials = run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': gym_name,
            'config': {
                **config
            },
            'checkpoint_freq': 50,
            'max_failures': 999,
            'stop': {
                'training_iteration': 1000,
            },
        }
    })
