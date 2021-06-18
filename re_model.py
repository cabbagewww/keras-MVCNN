import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input,Flatten, Dense,Dropout,Lambda
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

def reshape(a):
    dim = np.prod(a.get_shape().as_list()[1:])

    return tf.reshape(a, [-1, dim])

def f1(a):
    return tf.expand_dims(a, 0)

def f2(a):
    return tf.reduce_max(a, [0], name='view_pool')

def MVCNN_model(views_shape,n_views, n_classes):
# AlexNet
#     digit_input = Input(shape=views_shape)
#     conv1 = Conv2D(96,(11,11),strides=(4,4),padding='valid',activation='relu',use_bias=True)(digit_input)
#     pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(conv1)
#     conv2 = Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',use_bias=True)(pool1)
#     pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(conv2)
#     conv3 = Conv2D(384,(3,3),padding='same',activation='relu',use_bias=True)(pool2)
#     conv4 = Conv2D(384,(3,3),padding='same',activation='relu',use_bias=True)(conv3)
#     conv5 = Conv2D(256,(3,3),padding='same',activation='relu',use_bias=True)(conv4)
#     pool5 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(conv5)

#VGG-M
    digit_input = Input(shape=views_shape)
    conv1 = Conv2D(96,(7,7),strides=(2,2),padding='valid',activation='relu',use_bias=True)(digit_input)
    pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(conv1)
    conv2 = Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',use_bias=True)(pool1)
    pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(conv2)
    conv3 = Conv2D(512,(3,3),padding='same',activation='relu',use_bias=True)(pool2)
    conv4 = Conv2D(512,(3,3),padding='same',activation='relu',use_bias=True)(conv3)
    conv5 = Conv2D(512,(3,3),padding='same',activation='relu',use_bias=True)(conv4)
    pool5 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(conv5)

#ResNet
    # digit_input = Input(shape=views_shape)
    # pool5 = ResNet50(include_top=False,input_shape=views_shape)(digit_input)
    # CNN1_model = Model(digit_input,pool5)

    digit_inputs = []
    for i in range(n_views):

        digit_input = Input(shape=views_shape)
        digit_inputs.append(digit_input)
        CNN_out = CNN1_model(digit_inputs[i])
        CNN_out = Lambda(reshape)(CNN_out)
#         print(digit_inputs[i].shape)
#         print(CNN_out.shape)
        if i == 0:
            CNN_all = Lambda(f1)(CNN_out)
        else:
            cnn_out = Lambda(f1)(CNN_out)
            CNN_all = keras.layers.concatenate([CNN_all,cnn_out],axis=0)

    pool_vp = Lambda(f2)(CNN_all)
    dense1 = Dense(1024,activation='relu',use_bias=True)(pool_vp)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(512,activation='relu',use_bias=True)(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    out_put = Dense(n_classes,activation=None,use_bias=True)(dropout2)


    model_out = Model(digit_inputs, out_put)
    return model_out