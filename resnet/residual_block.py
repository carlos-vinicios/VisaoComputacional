import tensorflow as tf

def block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
  """A residual block.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 #always channels last

  if conv_shortcut:
    shortcut = tf.keras.layers.Conv2D(
        4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
    shortcut = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = tf.keras.layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

  x = tf.keras.layers.Conv2D(
      filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

  x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
  x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
  return x

def stack_block(x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.

  Returns:
    Output tensor for the stacked blocks.
  """
  x = block(x, filters, stride=stride1, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
  return x