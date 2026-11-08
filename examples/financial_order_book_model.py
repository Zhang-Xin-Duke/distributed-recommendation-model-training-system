"""DeepLOB-style three-class model and flat-weight interface for a parameter service."""
import math

import tensorflow as tf


def build_order_book_model(window=100):
    layers = tf.keras.layers
    inputs = layers.Input((window, 40, 1), name='order_book')
    x = inputs
    for width, stride in ((2, 2), (2, 2), (10, 1)):
        x = layers.Conv2D(32, (1, width), strides=(1, stride))(x)
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        for _ in range(2):
            x = layers.Conv2D(32, (4, 1), padding='same')(x)
            x = layers.LeakyReLU(negative_slope=0.01)(x)
    branches = []
    for length in (3, 5):
        branch = layers.Conv2D(64, (1, 1), activation='relu')(x)
        branches.append(layers.Conv2D(64, (length, 1), padding='same', activation='relu')(branch))
    pooled = layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(x)
    branches.append(layers.Conv2D(64, (1, 1), activation='relu')(pooled))
    sequence = layers.Reshape((window, 192))(layers.Concatenate()(branches))
    hidden = layers.LSTM(64)(sequence)
    return tf.keras.Model(inputs, layers.Dense(3)(hidden), name='order_book_direction')


def flat_order_book_step(model, class_weights=(1., 1., 1.)):
    """Return initial weights and a step that differentiates only the supplied weights."""
    shapes = [tuple(variable.shape) for variable in model.trainable_variables]
    sizes = [math.prod(shape) for shape in shapes]
    initial = tf.concat([tf.reshape(variable, [-1]) for variable in model.trainable_variables], axis=0)
    state = [tf.convert_to_tensor(variable) for variable in model.non_trainable_variables]

    @tf.function(autograph=False)
    def step(weights, order_book, labels):
        with tf.GradientTape() as tape:
            tape.watch(weights)
            parameters = [tf.reshape(values, shape)
                          for values, shape in zip(tf.split(weights, sizes), shapes)]
            logits, _ = model.stateless_call(parameters, state, order_book, training=False)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(losses*tf.gather(class_weights,labels))
        return loss, logits, tape.gradient(loss, weights)

    return initial, step
