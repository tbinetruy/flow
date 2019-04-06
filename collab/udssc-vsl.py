"""UDSSC (S)ingle (A)gent (R)einforcement (L)earning."""

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers import IDMController, ContinuousRouter,\
    SumoCarFollowingController, SumoLaneChangeController
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from flow.envs.minicity_env import MinicityEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MinicityScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import MinicityRouter

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env

import json
import numpy as np
seed=204
np.random.seed(seed)

# time horizon of a single rollout
HORIZON = 1000
# number of parallel workers
N_CPUS = 6
# number of rollouts per training iteration
N_ROLLOUTS = N_CPUS*1

vehicles = Vehicles()
vehicles.add(
    veh_id="manned",
    speed_mode=0b11111,
    lane_change_mode=0b011001010101,
    acceleration_controller=(SumoCarFollowingController, {}),
    lane_change_controller=(SumoLaneChangeController, {}),
    routing_controller=(MinicityRouter, {}),
    initial_speed=0,
    num_vehicles=50)

flow_params = dict(
    # name of the experiment
    exp_tag='udssc-vsl',

    # name of the flow environment the experiment is running on
    env_name='MinicityEnv',

    # name of the scenario class the experiment is running on
    scenario='MinicityScenario',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.1,
        render=False,
        seed=seed,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=1,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='uniform',
        min_gap=2.5,
    ),
)

def setup_exps():
    alg_run = 'DDPG'

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()

    config['num_workers'] = min(N_CPUS, N_ROLLOUTS)
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['horizon'] = HORIZON
    config['observation_filter'] = 'NoFilter'

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
    ray.init(num_cpus=N_CPUS + 1)
    trials = run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': gym_name,
            'config': {
                **config
            },
            'checkpoint_freq': 25,
            'max_failures': 999,
            'stop': {
                'training_iteration': 1000,
            },
            'num_samples': 3,
        },
    },
    resume=False,
    )
