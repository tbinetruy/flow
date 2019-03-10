"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import numpy as np
seed = 204
np.random.seed(seed)
import os
import sys

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
import gym

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1

Here the arguments are:
1 - the number of the checkpoint
"""

from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import linear, normc_initializer
import tensorflow as tf
from tensor2tensor.layers.common_attention import multihead_attention

VERBOSE = False

def residual_block(inputs):
    layer1 = tf.layers.conv2d(
        inputs=inputs,
        filters=8,
        kernel_size=[1, 2],
        strides=[1, 1],
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
    #initial_state = rnn_cell.zero_state(
    #    batch_size=inputs.get_shape().as_list()[0],
    #    dtype=tf.float32,
    #)
    inputs = tf.expand_dims(inputs, axis=1)
    prev_inputs = tf.expand_dims(prev_inputs, axis=1)
    outputs, _ = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=tf.concat([inputs, prev_inputs], 1),
        #initial_state=initial_state,
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

def relational_network(inputs, num_outputs):
    print('INPUT//////////////////////////////////////////')
    print(inputs)
    print('//////////////////////////////////////////INPUT')

    # Residual block for spatial processing
    _height = int(inputs.get_shape().as_list()[1])
    _width = int(inputs.get_shape().as_list()[2])
    channel = int(inputs.get_shape().as_list()[-1]/2)
    inputs = tf.log(inputs + 1)
    inputs, prev_inputs = tf.split(inputs, [channel, channel], axis=-1)
    curr_residual_outputs = residual_block(inputs)
    prev_residual_outputs = residual_block(prev_inputs)

    # Conv2DLSTM block for memeory processing
    conv2dlstm_outputs = conv2dlstm_block(
        curr_residual_outputs, prev_residual_outputs)

    # Flatten the conv2dlstm outputs
    conv2dlstm_outputs = tf.concat(
        [conv2dlstm_outputs[:,0,...], conv2dlstm_outputs[:,1,...]], -1)
    height = conv2dlstm_outputs.get_shape().as_list()[1]
    width = conv2dlstm_outputs.get_shape().as_list()[2]
    channel = conv2dlstm_outputs.get_shape().as_list()[3]
    flat_conv2dlstm_outputs = tf.reshape(
        conv2dlstm_outputs, [-1, height*width, channel])

    # MHDPA block for relational processing
    mhdpa_outputs = mhdpa_block(flat_conv2dlstm_outputs)
    channel = mhdpa_outputs.get_shape().as_list()[-1]

    # Optional: add additional mhdpa blocks
    #mhdpa_outputs = mhdpa_block(mhdpa_outputs)
    #mhdpa_outputs = mhdpa_block(mhdpa_outputs)

    # Flatten the mhdpa outputs
    flat_mhdpa_outputs = tf.layers.flatten(mhdpa_outputs)
    reshaped_mhdpa_outputs = tf.reshape(
        mhdpa_outputs, [-1, height, width, channel])

    # Feature layer to compute value function and policy logits
    feature_layer = flat_mhdpa_outputs
    logit_layer = tf.layers.conv2d_transpose(
        inputs=reshaped_mhdpa_outputs,
        filters=4,
        kernel_size=[1, 2],
        strides=[1, 1],
    )
    logit_layer = tf.layers.conv2d(
        inputs=logit_layer,
        filters=1,
        kernel_size=[1, 1],
    )
    flat_logit_layer = tf.layers.flatten(logit_layer)
    policy_logits = tf.layers.dense(flat_logit_layer, num_outputs)
    # Optional: use auto-regressive RNN

    print('OUTPUT/////////////////////////////////////////')
    print(policy_logits)
    print('/////////////////////////////////////////OUTPUT')
    return policy_logits, feature_layer

def perceptron_network(inputs, num_outputs):
    if VERBOSE:
        print('INPUT//////////////////////////////////////////')
        print(inputs)
        print('//////////////////////////////////////////INPUT')

    layer = tf.layers.flatten(inputs)
    if VERBOSE:
        print('LAYER1//////////////////////////////////////////')
        print(layer)
        print('//////////////////////////////////////////LAYER1')

    layer = tf.layers.dense(layer, 64)
    if VERBOSE:
        print('LAYER2//////////////////////////////////////////')
        print(layer)
        print('//////////////////////////////////////////LAYER2')

    feature_layer = tf.layers.dense(layer, 64)
    policy_logits = tf.layers.dense(feature_layer, num_outputs)
    if VERBOSE:
        print('OUTPUT/////////////////////////////////////////')
        print(feature_layer)
        print(policy_logits)
        print('/////////////////////////////////////////OUTPUT')

    return policy_logits, feature_layer

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

        policy_logits, feature_layer = relational_network(
            input_dict['obs'], num_outputs)
        return policy_logits, feature_layer

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


class _RLlibPreprocessorWrapper(gym.ObservationWrapper):
    """Adapts a RLlib preprocessor for use as an observation wrapper."""

    def __init__(self, env, preprocessor):
        super(_RLlibPreprocessorWrapper, self).__init__(env)
        self.preprocessor = preprocessor

        from gym.spaces.box import Box
        self.observation_space = Box(
            -1.0, 1.0, preprocessor.shape, dtype=np.float32)

    def observation(self, observation):
        return self.preprocessor.transform(observation)

def visualizer_rllib(args):
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 1

    flow_params = get_flow_params(config)

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(
        params=flow_params, version=0, render=False)
    register_env(env_name, create_env)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if (args.run and config_run):
        if (args.run != config_run):
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if (args.run):
        agent_cls = get_agent_class(args.run)
    elif (config_run):
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    # Recreate the scenario from the pickled parameters
    exp_tag = flow_params['exp_tag']
    net_params = flow_params['net']
    vehicles = flow_params['veh']
    initial_config = flow_params['initial']
    module = __import__('flow.scenarios', fromlist=[flow_params['scenario']])
    scenario_class = getattr(module, flow_params['scenario'])

    scenario = scenario_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    # Start the environment with the gui turned on and a path for the
    # emission file
    module = __import__('flow.envs', fromlist=[flow_params['env_name']])
    env_class = getattr(module, flow_params['env_name'])
    env_params = flow_params['env']
    if args.evaluate:
        env_params.evaluate = True
    sumo_params = flow_params['sumo']
    if args.no_render:
        sumo_params.render = False
    else:
        sumo_params.render = True
    sumo_params.emission_path = './test_time_rollout/'
    sumo_params.seed = seed

    _env = env_class(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario)
    _prep = ModelCatalog.get_preprocessor(_env, options={})
    env = _RLlibPreprocessorWrapper(_env, _prep)

    # Run the environment in the presence of the pre-trained RL agent for the
    # requested number of time steps / rollouts
    rets = []
    final_outflows = []
    mean_speed = []
    for i in range(args.num_rollouts):
        vel = []
        state = env.reset()
        ret = 0
        for _ in range(env_params.horizon):
            vehicles = env.unwrapped.vehicles
            vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            ret += reward
            if done:
                break
        rets.append(ret)
        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        mean_speed.append(np.mean(vel))
        print('Round {}, Return: {}'.format(i, ret))
    print('Average, std return: {}, {}'.format(np.mean(rets), np.std(rets)))
    print('Average, std speed: {}, {}'.format(np.mean(mean_speed),
                                              np.std(mean_speed)))
    print('Average, std outflow: {}, {}'.format(np.mean(final_outflows),
                                                np.std(final_outflows)))

    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.emission_to_csv:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(scenario.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        emission_to_csv(emission_path)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--emission_to_csv',
        action='store_true',
        help='Specifies whether to convert the emission file '
             'created by sumo into a csv file')
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Specifies whether to visualize the results')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    visualizer_rllib(args)
