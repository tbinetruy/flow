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
    IntersectionScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import IntersectionRandomRouter
from flow.core.params import InFlows
import numpy as np
seed=204
np.random.seed(seed)

from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import linear, normc_initializer
import tensorflow as tf
from tensor2tensor.layers.common_attention import multihead_attention

# time horizon of a single rollout
HORIZON = 1000
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 1

def residual_block(inputs):
    layer1 = tf.layers.conv2d(
        inputs=inputs,
        filters=8,
        kernel_size=[4, 2],
        strides=[4, 1],
        padding='valid',
        activation=tf.nn.relu,
    )
    layer2 = tf.layers.batch_normalization(layer1)
    layer3 = tf.layers.conv2d(
        inputs=layer2,
        filters=8,
        kernel_size=[2, 2],
        strides=[1, 1],
        padding='same',
        activation=None,
    )
    layer4 = tf.layers.batch_normalization(layer3)
    layer5 = tf.layers.conv2d(
        inputs=layer4,
        filters=8,
        kernel_size=[2, 2],
        strides=[1, 1],
        padding='same',
        activation=None,
    )
    outputs = tf.nn.relu(layer2 + tf.layers.batch_normalization(layer5))
    return outputs

def conv2dlstm_block(inputs, prev_inputs):
    rnn_cell = tf.contrib.rnn.Conv2DLSTMCell(
        input_shape=inputs.get_shape().as_list()[1:],
        output_channels=32,
        kernel_shape=[3, 3],
    )
    initial_state = rnn_cell.zero_state(
        batch_size=inputs.get_shape().as_list()[0],
        dtype=tf.float32,
    )
    inputs = tf.expand_dims(inputs, axis=1)
    prev_inputs = tf.expand_dims(prev_inputs, axis=1)
    outputs, _ = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=tf.concat([inputs, prev_inputs], 1),
        initial_state=initial_state,
        dtype=tf.float32,
    )
    return outputs

def mhdpa_block(inputs):
    outputs = multihead_attention(
        query_antecedent=inputs,
        memory_antecedent=None,
        bias=None,
        total_key_depth=inputs.get_shape().as_list()[1],
        total_value_depth=inputs.get_shape().as_list()[1],
        output_depth=32,
        num_heads=4,
        dropout_rate=0,
    )
    return outputs

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

        # Set inputs
        inputs = input_dict['obs']
        print('INPUT/////////////////////////////////////////')
        print(input_dict['obs'])
        print(input_dict['obs'].get_shape().as_list())

        # Allocate or access cache variable to store states
        with tf.variable_scope('residual_block_cache', reuse=tf.AUTO_REUSE):
            prev_residual_outputs = tf.get_variable(
                name='prev_residual_outputs',
                shape=[N_ROLLOUTS, 4, 4, 8],
                initializer=tf.zeros_initializer,
                trainable=False,
            )
            residual_outputs = tf.get_variable(
                name='residual_outputs',
                shape=[N_ROLLOUTS, 4, 4, 8],
                initializer=tf.zeros_initializer,
                trainable=False,
            )

        # Residual block for spatial processing
        residual_outputs.assign(residual_block(inputs))

        # Conv2DLSTM block for memeory processing
        conv2dlstm_outputs = conv2dlstm_block(
            residual_outputs, prev_residual_outputs)

        # Cache residual outputs
        prev_residual_outputs.assign(residual_outputs)

        # Flatten the conv2dlstm outputs
        conv2dlstm_outputs = tf.concat(
            [conv2dlstm_outputs[:,0,...], conv2dlstm_outputs[:,1,...]], -1)
        batch = conv2dlstm_outputs.get_shape().as_list()[0]
        height = conv2dlstm_outputs.get_shape().as_list()[1]
        width = conv2dlstm_outputs.get_shape().as_list()[2]
        channel = conv2dlstm_outputs.get_shape().as_list()[3]
        flat_conv2dlstm_outputs = tf.reshape(
            conv2dlstm_outputs, [batch, height*width, channel])

        # MHDPA block for relational processing
        mhdpa_outputs = mhdpa_block(flat_conv2dlstm_outputs)

        ## Optional additional mhdpa blocks
        ##mhdpa_outputs = mhdpa_block(mhdpa_outputs)
        ##mhdpa_outputs = mhdpa_block(mhdpa_outputs)

        # Flatten the mhdpa outputs
        batch = mhdpa_outputs.get_shape().as_list()[0]
        flat_mhdpa_outputs = tf.reshape(mhdpa_outputs, [batch, -1])

        ## Feature layer to compute value function and policy logits
        feature_layer = flat_mhdpa_outputs
        policy_logits = tf.layers.dense(
            feature_layer,
            num_outputs
        )

        print('OUTPUT/////////////////////////////////////////')
        print(policy_logits)
        print(policy_logits.get_shape().as_list())

        return policy_logits, feature_layer

        #import tensorflow as tf
        #import tensorflow.contrib.slim as slim
        #from ray.rllib.models.model import Model
        #from ray.rllib.models.misc import normc_initializer, get_activation_fn
        #from ray.rllib.utils.annotations import override

        #hiddens = options.get("fcnet_hiddens")
        #activation = get_activation_fn(options.get("fcnet_activation"))

        #with tf.name_scope("fc_net"):
        #    i = 1
        #    last_layer = tf.reshape(input_dict["obs"], [-1, 4])
        #    for size in hiddens:
        #        label = "fc{}".format(i)
        #        last_layer = slim.fully_connected(
        #            last_layer,
        #            size,
        #            weights_initializer=normc_initializer(1.0),
        #            activation_fn=activation,
        #            scope=label)
        #        i += 1
        #    label = "fc_out"
        #    output = slim.fully_connected(
        #        last_layer,
        #        num_outputs,
        #        weights_initializer=normc_initializer(0.01),
        #        activation_fn=None,
        #        scope=label)
        #    return output, last_layer

    def value_function(self):
        """Builds the value function output.

        This method can be overridden to customize the implementation of the
        value function (e.g., not sharing hidden layers).

        Returns:
            Tensor of size [BATCH_SIZE] for the value function.
        """
        return tf.reshape(
            linear(self.last_layer, 1, "value", normc_initializer(1.0)), [-1])

    def custom_loss(self, policy_loss, loss_inputs):
        """Override to customize the loss function used to optimize this model.

        This can be used to incorporate self-supervised losses (by defining
        a loss over existing input and output tensors of this model), and
        supervised losses (by defining losses over a variable-sharing copy of
        this model's layers).

        You can find an runnable example in examples/custom_loss.py.

        Arguments:
            policy_loss (Tensor): scalar policy loss from the policy graph.
            loss_inputs (dict): map of input placeholders for rollout data.

        Returns:
            Scalar tensor for the customized loss for this model.
        """
        return policy_loss

    def custom_stats(self):
        """Override to return custom metrics from your model.

        The stats will be reported as part of the learner stats, i.e.,
            info:
                learner:
                    model:
                        key1: metric1
                        key2: metric2

        Returns:
            Dict of string keys to scalar tensors.
        """
        return {}

ModelCatalog.register_custom_model('relational_model', RelationalModelClass)

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
    exp_tag='intersection',

    # name of the flow environment the experiment is running on
    env_name='IntersectionEnv',

    # name of the scenario class the experiment is running on
    scenario='IntersectionScenario',

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
                'training_iteration': 10000,
            },
            #'local_dir': '/mnt/d/Overflow/ray_results/',
            'num_samples': 1,
        },
    },
    resume='prompt',
    verbose=1,)
