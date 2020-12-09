import tensorflow as tf

def dense_block(x, blocks, name):
  """A dense block.

  Arguments:
    x: input tensor.
    blocks: integer, the number of building blocks.
    name: string, block label.

  Returns:
    Output tensor for the block.
  """
  for i in range(blocks):
    x = conv_block(x, 32, name=name + '_block' + str(i + 1))
  
  return x

def transition_block(x, reduction, name):
  """A transition block.

  Arguments:
    x: input tensor.
    reduction: float, compression rate at transition layers.
    name: string, block label.

  Returns:
    output tensor for the block.
  """
  bn_axis = 3
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
          x)
  x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
  x = tf.keras.layers.Conv2D(
      int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
      1,
      use_bias=False,
      name=name + '_conv')(
          x)
  x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
  return x

def conv_block(x, growth_rate, name):
  """A building block for a dense block.

  Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

  Returns:
    Output tensor for the block.
  """
  bn_axis = 3
  x1 = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
          x)
  x1 = tf.keras.layers.Activation('relu', name=name + '_0_relu')(x1)
  x1 = tf.keras.layers.Conv2D(
      4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
          x1)
  x1 = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
          x1)
  x1 = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x1)
  x1 = tf.keras.layers.Conv2D(
      growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
          x1)
  x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
  return x