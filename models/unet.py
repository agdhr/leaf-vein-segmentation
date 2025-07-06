"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
Authors: Olaf Ronneberger, Philipp Fischer, Thomas Broxauthor
https://github.com/decouples/Unet/tree/master
https://arxiv.org/abs/1505.04597
"""

import os
from turtle import up
import numpy as np
import tensorflow as tf

from keras import models, layers, optimizers#, callbacks
from keras import backend as K
from torch import mode
from model.metrics import iou, jacard_coef, dice_coef


def unet_2d(input_size, n_class=1,):
    inputs = layers.Input(input_size)
	# 网络结构定义，数据处理的时候已经转化为灰度图了
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #print ("conv1 shape:",conv1.shape)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    #print (`"conv1 shape:",conv1.shape)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    #print ("pool1 shape:",pool1.shape)

    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #print ("conv2 shape:",conv2.shape)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #print ("conv2 shape:",conv2.shape)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    #print ("pool2 shape:",pool2.shape)

    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #print ("conv3 shape:",conv3.shape)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #print ("conv3 shape:",conv3.shape)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    #print ("pool3 shape:",pool3.shape)

    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(drop5))
    # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = layers.Concatenate(axis=3)([drop4, up6])
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv6))
    # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    merge7 = layers.Concatenate(axis=3)([conv3, up7])
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv7))
    # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    merge8 = layers.Concatenate(axis=3)([conv2,up8])
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv8))
    # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    merge9 = layers.Concatenate(axis=3)([conv1,up9])
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = layers.Conv2D(n_class, 1, activation = 'sigmoid')(conv9)
	# [batch, 512, 512, 1]

    model = models.Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = optimizers.Adam(learning_rate = 1e-4), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy',iou, jacard_coef,dice_coef])
    return model

def bn_act(x, act=True):
    x = layers.BatchNormalization()(x)
    if act == True:
        x = layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    x = bn_act(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    return x

def conv2_block(x, filters):
    x = conv_block(x, filters, kernel_size=(3, 3), padding='same', strides=1)
    x = conv_block(x, filters, kernel_size=(3, 3), padding='same', strides=1)
    return x

def upsample_block(x, skip, filters):
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.Concatenate(axis=3)([skip, x])
    return conv2_block(x, filters)

def UNet(input_size, n_class=1):
    f = [32, 64, 128, 256, 512, 1024]
    # Define the input layer
    inputs = layers.Input(input_size)

    # Encoder
    c1 = conv2_block(inputs, f[0])
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv2_block(p1, f[1])
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv2_block(p2, f[2])
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv2_block(p3, f[3])
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = conv_block(p4, f[4])

    # Decoder
    u6 = upsample_block(c5, c4, f[3])
    u7 = upsample_block(u6, c3, f[2])
    u8 = upsample_block(u7, c2, f[1])
    u9 = upsample_block(u8, c1, f[0])

    outputs = layers.Conv2D(n_class, 1, activation='sigmoid' if n_class == 1 else 'softmax')(u9)

    model = models.Model(inputs, outputs)
    model.compile(optimizer = optimizers.Adam(learning_rate = 1e-4), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy',iou, jacard_coef,dice_coef])
    return model

def stem(x, filters, kernel_size=(3,3), strides=1, padding="same"):
    conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = layers.Conv2D(filters, kernel_size=(1,1), strides=strides, padding=padding)(x)
    shortcut = bn_act(shortcut, act=False)

    output = layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3,3), strides=1, padding="same"):
    res = conv_block(x, filters, kernel_size=kernel_size, strides=strides, padding=padding)
    res = conv_block(res, filters, kernel_size=kernel_size, strides=1, padding=padding)
    
    shortcut = layers.Conv2D(filters, kernel_size=(1,1), strides=strides, padding=padding)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = layers.Add()([shortcut, res])
    #output = layers.Activation('relu')
    return output

def upsample_concat_block(x, skip):
    u = layers.UpSampling2D((2,2))(x)
    c = layers.Concatenate(axis=3)([u, skip])
    return c

def ResUNet(input_shape, n_class=1):
    """
    The ResUNet-a model
    model_build_func(input_shape, n_labels)
    """
    f = [32, 64, 128, 256, 512, 1024]
    inputs = layers.Input(input_shape)

    # Encoder
    e0 = inputs
    e1 = stem(e0, 32)
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    # Bridge
    b0 = conv_block(e5, f[5], strides=1)
    b1 = conv_block(b0, f[5], strides=1)

    # Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = layers.Conv2D(n_class, 1, activation='sigmoid' if n_class == 1 else 'softmax')(d4)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = optimizers.Adam(learning_rate = 1e-4), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy',iou, jacard_coef,dice_coef])
    return model

def conv_block_for_att(x, size, filter_size, dropout, batch_norm=False):
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    return conv 

def gating_signal(x, filters, batch_norm=True):
    x = layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    if batch_norm:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    return x

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape
    #(None, 256,256,6), if specified axis=3 and rep=2.

    #return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return layers.Lambda(lambda x: tf.repeat(x, repeats=rep, axis=3))(tensor)

#from tensorflow.keras.layers import Lambda
#def attention_block(x, gating, inter_shape):
#    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(1, 1))(x)

#    phi_g = layers.Conv2D(inter_shape, (2, 2), strides=(1, 1))(gating)

    # Resize phi_g to match theta_x spatial dims
#    phi_g_resized = Lambda(lambda t: tf.image.resize(t, tf.shape(theta_x)[1:3]))(phi_g)

#    f = layers.Activation('relu')(layers.add([theta_x, phi_g_resized]))

#    psi_f = layers.Conv2D(1, (1, 1), strides=(1, 1))(f)

#    rate = layers.Activation('sigmoid')(psi_f)

#    att_x = layers.multiply([x, rate])
#    return att_x

def attention_block(x, gating, inter_shape):
    shape_x = x.shape
    shape_g = gating.shape

    # Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = theta_x.shape

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = sigmoid_xg.shape
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def Attention_UNet(input_shape, n_class=1, dropout_rate=0, batch_norm = True):
    f = [32, 64, 128, 256, 512, 1024]
    inputs = layers.Input(input_shape)
    filter_size = 3

    # Encoder
    c1 = conv_block_for_att(inputs, f[0], filter_size, dropout_rate, batch_norm)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_for_att(p1, f[1], filter_size, dropout_rate, batch_norm)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_for_att(p2, f[2], filter_size, dropout_rate, batch_norm)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_for_att(p3, f[3], filter_size, dropout_rate, batch_norm)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = conv_block_for_att(p4, f[4], filter_size, dropout_rate, batch_norm)

    # Decoder with attention blocks
    g6 = gating_signal(c5, f[3])
    a6 = attention_block(c4, g6, f[3])
    u6 = upsample_concat_block(c5, a6)
    up_conv6 = conv_block_for_att(u6, f[3], filter_size, dropout_rate, batch_norm)

    g7 = gating_signal(up_conv6, f[2])
    a7 = attention_block(c3, g7, f[2])
    u7 = upsample_concat_block(u6, a7)
    up_conv7 = conv_block_for_att(u7, f[2], filter_size, dropout_rate, batch_norm)

    g8 = gating_signal(up_conv7, f[1])
    a8 = attention_block(c2, g8, f[1])
    u8 = upsample_concat_block(up_conv7, a8)
    up_conv8 = conv_block_for_att(u8, f[1], filter_size, dropout_rate, batch_norm)

    g9 = gating_signal(up_conv8, f[0])
    a9 = attention_block(c1, g9, f[0])
    u9 = upsample_concat_block(up_conv8, a9)
    up_conv9 = conv_block_for_att(u9, f[0], filter_size, dropout_rate, batch_norm)

    outputs = layers.Conv2D(n_class, 1, activation='sigmoid' if n_class == 1 else 'softmax')(up_conv9)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = optimizers.Adam(learning_rate = 1e-4), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy',iou, jacard_coef,dice_coef])
    
    return model

def Attention_ResUNet(input_shape, n_class=1):
    f = [32, 64, 128, 256, 512, 1024]
    inputs = layers.Input(input_shape)
    # Encoder
    c1 = residual_block(inputs, f[0])
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = residual_block(p1, f[1])
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = residual_block(p2, f[2])
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = residual_block(p3, f[3])
    p4 = layers.MaxPooling2D((2, 2))(c4)
    # Bottleneck
    c5 = residual_block(p4, f[4])
    # Decoder with attention blocks
    g6 = gating_signal(c5, f[3])
    a6 = attention_block(c4, g6, f[3])
    u6 = upsample_concat_block(c5, a6)
    up_conv6 = residual_block(u6, f[3])
    g7 = gating_signal(up_conv6, f[2])
    a7 = attention_block(c3, g7, f[2])
    u7 = upsample_concat_block(up_conv6, a7)
    up_conv7 = residual_block(u7, f[2])
    g8 = gating_signal(up_conv7, f[1])
    a8 = attention_block(c2, g8, f[1])
    u8 = upsample_concat_block(up_conv7, a8)
    up_conv8 = residual_block(u8, f[1])
    g9 = gating_signal(up_conv8, f[0])
    a9 = attention_block(c1, g9, f[0])
    u9 = upsample_concat_block(up_conv8, a9)
    up_conv9 = residual_block(u9, f[0])
    outputs = layers.Conv2D(n_class, 1, activation='sigmoid' if n_class == 1 else 'softmax')(up_conv9)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = optimizers.Adam(learning_rate = 1e-4), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy',iou, jacard_coef,dice_coef])
    return model