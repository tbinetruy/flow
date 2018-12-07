"""Runs the environments located in flow/benchmarks.
The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the PPO algorithm in rllib
and utilizes the hyper-parameters specified in:
Proximal Policy Optimization Algorithms by Schulman et. al.
"""
import json

import ray
from ray.rllib.agents.agent import get_agent_class
from ray import tune
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# use this to specify the environment to run
from flow.benchmarks.merge0 import flow_params

# number of rollouts per training iteration
N_ROLLOUTS = 30
# number of parallel workers
N_CPUS = 15

if __name__ == "__main__":
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = 600 * N_ROLLOUTS
    #config['kl_coeff'] = tune.grid_search([0.002, 0.2])
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [100, 50, 25]})
    config['lr'] = tune.grid_search([5e-4, 5e-5])
    config["use_gae"] = True
    config["lambda"] = 0.97
    #config["num_sgd_iter"] = tune.grid_search([10, 30])
    config['horizon'] = 600
    config['observation_filter'] = 'NoFilter'

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)
    ray.init(redis_address="localhost:6379")
    run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 50,
            'stop': {
                'training_iteration': 600
            },
            'config': config,
            'upload_dir': 's3://eugene.experiments/singleagent_merge_v1',
            'num_samples': 2
        },

    })
