"""Intersection (S)ingle (A)gent (R)einforcement (L)earning."""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers import IDMController, ContinuousRouter,\
    SumoCarFollowingController, SumoLaneChangeController
from flow.controllers.routing_controllers import IntersectionRouter
from flow.envs.intersection_env import IntersectionEnv, \
    ADDITIONAL_ENV_PARAMS
from flow.scenarios.intersection import \
    SoftIntersectionScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import IntersectionRandomRouter
from flow.core.params import InFlows
import numpy as np
seed=204
np.random.seed(seed)

from ray.rllib.models import ModelCatalog, Model
import tensorflow as tf

class RelationalModelClass(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].

        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:

        Examples:
            >>> print(input_dict)
            {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
             'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
             'is_training': <tf.Tensor shape=(), dtype=bool>,
             'obs': OrderedDict([
                ('sensors', OrderedDict([
                    ('front_cam', [
                        <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>,
                        <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>]),
                    ('position', <tf.Tensor shape=(?, 3) dtype=float32>),
                    ('velocity', <tf.Tensor shape=(?, 3) dtype=float32>)]))])}
        """

        #layer1 = slim.fully_connected(input_dict["obs"], 64, ...)
        #layer2 = slim.fully_connected(layer1, 64, ...)
        #...
        conv = tf.layers.conv2d(
            inputs=input_dict["obs"],
            filters=8,
            kernel_size=[4, 2],
            padding="valid",
            activation=tf.nn.relu
        )
        return layerN, layerN_minus_1

    def value_function(self):
        """Builds the value function output.

        This method can be overridden to customize the implementation of the
        value function (e.g., not sharing hidden layers).

        Returns:
            Tensor of size [BATCH_SIZE] for the value function.
        """
        return tf.reshape(
            linear(self.last_layer, 1, "value", normc_initializer(1.0)), [-1])

    def custom_loss(self, policy_loss):
        """Override to customize the loss function used to optimize this model.

        This can be used to incorporate self-supervised losses (by defining
        a loss over existing input and output tensors of this model), and
        supervised losses (by defining losses over a variable-sharing copy of
        this model's layers).

        You can find an runnable example in examples/custom_loss.py.

        Arguments:
            policy_loss (Tensor): scalar policy loss from the policy graph.

        Returns:
            Scalar tensor for the customized loss for this model.
        """
        return policy_loss


ModelCatalog.register_custom_model('relational_model', RelationalModelClass)

# time horizon of a single rollout
HORIZON = 1000
# number of rollouts per training iteration
N_ROLLOUTS = 6*2
# number of parallel workers
N_CPUS = 6

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()

# We place 40 autonomous vehicles in the network
vehicles = Vehicles()

# Add mixed-autonomy traffic
insertion_prob = 0.1
autonomy_percent = 1.0#0.5
prob_table = {
    'manned': (1 - autonomy_percent)*insertion_prob,
    'autonomous': autonomy_percent*insertion_prob,
}
inflow = InFlows()
for type in ['autonomous']:#['manned', 'autonomous']:
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
    env_name='IntersectionEnv',

    # name of the scenario class the experiment is running on
    scenario='SoftIntersectionScenario',

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
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
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
    grad_free = False
    if grad_free:
        alg_run = 'ES'
    else:
        alg_run = 'A2C'

    use_custom_model = True

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()

    config['num_workers'] = min(N_CPUS, N_ROLLOUTS)
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['horizon'] = HORIZON
    if grad_free:
        # pass
        config['episodes_per_batch'] = N_ROLLOUTS
        config['eval_prob'] = 0.05
        config['noise_stdev'] = 0.05
        config['stepsize'] = 0.02
        config['clip_actions'] = False
        config['observation_filter'] = 'NoFilter'
    else:
        pass
        # config["use_gae"] = True
        # config["lambda"] = 0.97
        config["lr"] = 5e-5
        # config["vf_clip_param"] = 1e6
        # config["num_sgd_iter"] = 10
        # #config["model"]["fcnet_hiddens"] = [100, 50, 25]
        config["observation_filter"] = "NoFilter"

        if use_custom_model:
            config['model'] = {
                'custom_model': 'relational_model',
                'custom_options': {},
            }

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
            'checkpoint_freq': 50,
            'max_failures': 999,
            'stop': {
                'training_iteration': 1000,
            },
            'local_dir': '/mnt/d/Overflow/ray_results/',
            'num_samples': 1,
        },
    },
    resume=False,#'prompt',
    verbose=1,)