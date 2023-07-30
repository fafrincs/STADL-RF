#! /usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import \
    Dense, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D, Input, Activation, MaxPooling1D, MaxPooling2D, \
    Conv1D, Conv2D, BatchNormalization, LSTM, Flatten, ELU, AveragePooling1D, Permute
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import cvnn.layers as complex_layers
import cvnn.activations
from tensorflow.keras.losses import Loss, categorical_crossentropy
from keras.optimizers import RMSprop


class ComplexAverageCrossEntropy(Loss):

    def call(self, y_true, y_pred):
        real_loss = categorical_crossentropy(y_true, tf.math.real(y_pred))
        if y_pred.dtype.is_complex:
            imag_loss = categorical_crossentropy(y_true, tf.math.imag(y_pred))
        else:
            imag_loss = real_loss
        return (real_loss + imag_loss) / 2



def createSB_cart(inp_shape, classes_num=6, emb_size=64, weight_decay=1e-4, classification=False):

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 2, 2, 2, 2]
    conv_stride_size = ['None', 2, 2, 2, 2]
    pool_stride_size = ['None', 1, 1, 1, 1]
    pool_size = ['None', 2, 2, 2, 2]

    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(inp_shape, dtype='complex64'))
    model.add(complex_layers.ComplexConv1D(filter_num[1], kernel_size[1], strides=conv_stride_size[1], activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(pool_size[1], strides=pool_stride_size[1]))

    model.add(complex_layers.ComplexConv1D(filter_num[2], kernel_size[2], strides=conv_stride_size[2], activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(pool_size[2], strides=pool_stride_size[2]))

    model.add(complex_layers.ComplexConv1D(filter_num[3], kernel_size[3], strides=conv_stride_size[3], activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(pool_size[3], strides=pool_stride_size[3]))

    model.add(complex_layers.ComplexConv1D(filter_num[4], kernel_size[4], strides=conv_stride_size[4], activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(pool_size[4], strides=pool_stride_size[4]))

    # Add more convolutional layers if desired
    model.add(complex_layers.ComplexConv1D(filter_num[4], kernel_size[4], strides=conv_stride_size[4], activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(pool_size[4], strides=pool_stride_size[4]))

    model.add(complex_layers.ComplexFlatten())

    if classification:
        model.add(complex_layers.ComplexDense(classes_num, activation='softmax_real_with_abs'))
    else:
        model.add(complex_layers.ComplexDense(emb_size, activation='linear',
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                              bias_regularizer=tf.keras.regularizers.l2(weight_decay)))

    return model


def test_run(model):
    model.compile(optimizer='RMSprop', loss=ComplexAverageCrossEntropy(), metrics=['accuracy'])
    # opt = tf.keras.optimizers.RMSprop(clipnorm=1.0)
    # model.compile(loss=ComplexAverageCrossEntropy(), metrics=["accuracy"], optimizer=opt)


def test():
    modelTypes = ['complex']
    reluType = ['cart']
    NUM_CLASS = 6
    signal = True
    inp_shape = (1, 288)
    emb_size = 64
    for reluType in modelTypes:
        model = create_model(reluType, inp_shape, NUM_CLASS, emb_size, classification=True)
        try:
            test_run(model)
        except Exception as e:
            print(e)
    print('all done!') if signal else print('test failed')


if __name__ == "__main__":
    test()
