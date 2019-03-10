import tensorflow as tf
from tensor2tensor.layers.common_attention import multihead_attention
import numpy as np

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

inputs_ph = tf.placeholder(tf.float32, shape=(None, 16, 5, 2))

# Residual block for spatial processing
_height = int(inputs_ph.get_shape().as_list()[1])
_width = int(inputs_ph.get_shape().as_list()[2])
channel = int(inputs_ph.get_shape().as_list()[-1]/2)
inputs_ph = tf.log(inputs_ph + 1)
inputs, prev_inputs = tf.split(inputs_ph, [channel, channel], axis=-1)
residual_outputs = residual_block(inputs)
prev_residual_outputs = residual_block(prev_inputs)

# Conv2DLSTM block for memeory processing
conv2dlstm_outputs = conv2dlstm_block(
    residual_outputs, prev_residual_outputs)

# Flatten the conv2dlstm outputs
conv2dlstm_outputs = tf.concat(
    [conv2dlstm_outputs[:,0,...], conv2dlstm_outputs[:,1,...]], -1)
batch = conv2dlstm_outputs.get_shape().as_list()[0]
height = conv2dlstm_outputs.get_shape().as_list()[1]
width = conv2dlstm_outputs.get_shape().as_list()[2]
channel = conv2dlstm_outputs.get_shape().as_list()[3]
flat_conv2dlstm_outputs = tf.reshape(
    conv2dlstm_outputs, [-1, height*width, channel])

# MHDPA block for relational processing
mhdpa_outputs = mhdpa_block(flat_conv2dlstm_outputs)
print(mhdpa_outputs)

## Optional: add additional mhdpa blocks
##mhdpa_outputs = mhdpa_block(mhdpa_outputs)
##mhdpa_outputs = mhdpa_block(mhdpa_outputs)

# Reshape the mhdpa outputs
flat_mhdpa_outputs = tf.layers.flatten(mhdpa_outputs)
reshaped_mhdpa_outputs = tf.reshape(
    mhdpa_outputs, [-1, height, width, channel])

# Feature layer to compute value function and policy logits
feature_layer = flat_mhdpa_outputs
logit_layer = tf.layers.conv2d_transpose(
    inputs=reshaped_mhdpa_outputs,
    filters=4,
    kernel_size=[4, 2],
    strides=[4, 1],
)
logit_layer = tf.layers.conv2d(
    inputs=logit_layer,
    filters=1,
    kernel_size=[1, 1],
)
flat_logit_layer = tf.layers.flatten(logit_layer)
policy_logits = tf.layers.dense(flat_logit_layer, 4)
# Optional: use auto-regressive RNN

with tf.Session() as sess:
    writer = tf.summary.FileWriter('relational_model', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(policy_logits,
        feed_dict={inputs_ph: np.zeros((8, 16, 5, 2))})
    writer.close()
