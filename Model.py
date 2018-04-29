import tensorflow as tf

he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False)

def convolutional_layer(input, layer, window, stride, resnet):
    bias = window[-1]
    kernel = tf.get_variable('kernel', window, initializer=he_init)
    biases = tf.get_variable('biases', bias, initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(input, kernel, strides=[1, stride, stride, 1], padding='SAME')

    res4 = window[-1]
    res3 = window[-2]

    if resnet == True:
        if stride == 2:

            kernelres = tf.get_variable('kernelres', [1, 1, res3, res4], initializer=he_init)
            resinput = tf.nn.conv2d(input, kernelres, strides=[1, 2, 2, 1], padding='SAME')

            conv = conv + resinput

        elif res4 != res3 and stride != 2:

            kernelres = tf.get_variable('kernelres', [1, 1, res3, res4], initializer=he_init)
            resinput = tf.nn.conv2d(input, kernelres, strides=[1, 1, 1, 1], padding='SAME')

            conv = conv + resinput

        else:
            conv = conv + input

    conv = tf.nn.relu(conv + biases)
    print("conv" + str(layer))
    print(conv.shape)

    return conv


def pooling_layer(input, layer, kernel, stride):
    pool = tf.nn.max_pool(input, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME')
    print("pool" + str(layer))
    print(pool.shape)

    return pool