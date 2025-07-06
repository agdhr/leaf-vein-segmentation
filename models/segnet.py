"""
code adapted from:
[1] https://github.com/ykamikawa/tf-keras-SegNet
[2] https://github.com/danielenricocahall/Keras-SegNet/tree/master/segnet
"""

from keras import layers, models, optimizers
from model.metrics import iou, jacard_coef, dice_coef
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

from keras.layers import Layer


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = tf.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = tf.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3])

        ret = tf.scatter_nd(
            tf.expand_dims(tf.reshape(mask, [-1]), axis=1),  # Indices as [N, 1]
            tf.reshape(updates, [-1]),                       # Values flattened
            [tf.reduce_prod(output_shape)]                  # Output shape as scalar
            )

        input_shape = updates.shape
        out_shape = [-1,
                     input_shape[1] * self.size[0],
                     input_shape[2] * self.size[1],
                     input_shape[3]]
        return tf.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )
    
def segnet(input_shape, n_labels=1, kernel=3, pool_size=(2, 2), output_mode="sigmoid"):
    # encoder
    inputs = layers.Input(shape=input_shape)

    conv_1 = layers.Conv2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = layers.BatchNormalization()(conv_1)
    conv_1 = layers.Activation("relu")(conv_1)
    conv_2 = layers.Conv2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = layers.BatchNormalization()(conv_2)
    conv_2 = layers.Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = layers.Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = layers.BatchNormalization()(conv_3)
    conv_3 = layers.Activation("relu")(conv_3)
    conv_4 = layers.Conv2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = layers.BatchNormalization()(conv_4)
    conv_4 = layers.Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = layers.Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = layers.BatchNormalization()(conv_5)
    conv_5 = layers.Activation("relu")(conv_5)
    conv_6 = layers.Conv2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = layers.BatchNormalization()(conv_6)
    conv_6 = layers.Activation("relu")(conv_6)
    conv_7 = layers.Conv2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = layers.BatchNormalization()(conv_7)
    conv_7 = layers.Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = layers.Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = layers.BatchNormalization()(conv_8)
    conv_8 = layers.Activation("relu")(conv_8)
    conv_9 = layers.Conv2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = layers.BatchNormalization()(conv_9)
    conv_9 = layers.Activation("relu")(conv_9)
    conv_10 = layers.Conv2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = layers.BatchNormalization()(conv_10)
    conv_10 = layers.Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = layers.Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = layers.BatchNormalization()(conv_11)
    conv_11 = layers.Activation("relu")(conv_11)
    conv_12 = layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = layers.BatchNormalization()(conv_12)
    conv_12 = layers.Activation("relu")(conv_12)
    conv_13 = layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = layers.BatchNormalization()(conv_13)
    conv_13 = layers.Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = layers.Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = layers.BatchNormalization()(conv_14)
    conv_14 = layers.Activation("relu")(conv_14)
    conv_15 = layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = layers.BatchNormalization()(conv_15)
    conv_15 = layers.Activation("relu")(conv_15)
    conv_16 = layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = layers.BatchNormalization()(conv_16)
    conv_16 = layers.Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = layers.Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = layers.BatchNormalization()(conv_17)
    conv_17 = layers.Activation("relu")(conv_17)
    conv_18 = layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = layers.BatchNormalization()(conv_18)
    conv_18 = layers.Activation("relu")(conv_18)
    conv_19 = layers.Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = layers.BatchNormalization()(conv_19)
    conv_19 = layers.Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = layers.Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = layers.BatchNormalization()(conv_20)
    conv_20 = layers.Activation("relu")(conv_20)
    conv_21 = layers.Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = layers.BatchNormalization()(conv_21)
    conv_21 = layers.Activation("relu")(conv_21)
    conv_22 = layers.Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = layers.BatchNormalization()(conv_22)
    conv_22 = layers.Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = layers.Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = layers.BatchNormalization()(conv_23)
    conv_23 = layers.Activation("relu")(conv_23)
    conv_24 = layers.Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = layers.BatchNormalization()(conv_24)
    conv_24 = layers.Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = layers.Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = layers.BatchNormalization()(conv_25)
    conv_25 = layers.Activation("relu")(conv_25)

    conv_26 = layers.Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    #conv_26 = layers.BatchNormalization()(conv_26)
    #conv_26 = layers.Reshape(
    #    (input_shape[0] * input_shape[1], n_labels),
    #    input_shape=(input_shape[0], input_shape[1], n_labels),
    #)(conv_26)

    outputs = layers.Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = models.Model(inputs=inputs, outputs=outputs, name="SegNet")
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])  
    
    return model


def conv_bn_relu(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', 
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)

def SegNet(input_shape, n_classes=1, filter_num=32, pool_size=(2, 2)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = conv_bn_relu(inputs, filter_num)
    x = conv_bn_relu(x, filter_num)
    x, mask_1 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = conv_bn_relu(x, filter_num * 2)
    x = conv_bn_relu(x, filter_num * 2)
    x, mask_2 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = conv_bn_relu(x, filter_num * 4)
    x = conv_bn_relu(x, filter_num * 4)
    x = conv_bn_relu(x, filter_num * 4)
    x, mask_3 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = conv_bn_relu(x, filter_num * 8)
    x = conv_bn_relu(x, filter_num * 8)
    x = conv_bn_relu(x, filter_num * 8)
    x, mask_4 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = conv_bn_relu(x, filter_num * 8)
    x = conv_bn_relu(x, filter_num * 8)
    x = conv_bn_relu(x, filter_num * 8)
    x, mask_5 = MaxPoolingWithArgmax2D(pool_size)(x)

    # Decoder (with channel alignment)
    def unpool_and_conv(x, mask, out_filters):
        x = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = MaxUnpooling2D(pool_size)([x, mask])
        x = conv_bn_relu(x, out_filters)
        x = conv_bn_relu(x, out_filters)
        return x

    x = unpool_and_conv(x, mask_5, filter_num * 8)
    x = conv_bn_relu(x, filter_num*8)
    x = unpool_and_conv(x, mask_4, filter_num * 8)
    x = conv_bn_relu(x, filter_num*8)
    x = unpool_and_conv(x, mask_3, filter_num * 4)
    x = conv_bn_relu(x, filter_num*4)
    x = unpool_and_conv(x, mask_2, filter_num * 2)
    x = unpool_and_conv(x, mask_1, filter_num)

    output = layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', jacard_coef, dice_coef])
    return model

def residual2_block(x, filters):
    res = conv_bn_relu(x, filters)
    res = conv_bn_relu(res, filters)
    
    shortcut = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    out = layers.add([res, shortcut])
    #out = layers.Activation('relu')
    return out

def residual3_block(x, filters):
    res = conv_bn_relu(x, filters)
    res = conv_bn_relu(res, filters)
    res = conv_bn_relu(res, filters)
    
    shortcut = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    out = layers.add([res, shortcut])
    #out = layers.Activation('relu')
    return out

def ResSegNet(input_shape, n_classes=1, pool_size=(2, 2)):
    f = [32, 64, 128, 256, 512, 1024]
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = residual2_block(inputs, f[0])
    p1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(c1)
    c2 = residual2_block(p1, f[1])
    p2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(c2)
    c3 = residual2_block(p2, f[2])
    p3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(c3)
    c4 = residual2_block(p3, f[3])
    p4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(c4)
    c5 = residual2_block(p4, f[4])
    p5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(c5)

    # Decoder (with channel alignment)
    def unpool_and_res3(x, mask, out_filters):
        x = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = MaxUnpooling2D(pool_size)([x, mask])
        x = residual3_block(x, out_filters)
        return x
    def unpool_and_res2(x, mask, out_filters):
        x = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = MaxUnpooling2D(pool_size)([x, mask])
        x = residual2_block(x, out_filters)
        return x
    
    # Decoder
    d1 = unpool_and_res2(p5, mask_5, f[4])
    d2 = unpool_and_res2(d1, mask_4, f[3])
    d3 = unpool_and_res2(d2, mask_3, f[2])
    d4 = unpool_and_res2(d3, mask_2, f[1])
    d5 = unpool_and_res2(d4, mask_1, f[0])

    output = layers.Conv2D(n_classes, (1, 1), activation='sigmoid' if n_classes == 1 else 'softmax')(d5)
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer = optimizers.Adam(learning_rate = 1e-4), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy', jacard_coef,dice_coef])
    return model

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

def attention_block1(x, gating, inter_shape):
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(1, 1))(x)
    phi_g = layers.Conv2D(inter_shape, (2, 2), strides=(1, 1))(gating)
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, (1, 1), strides=(1, 1))(f)
    rate = layers.Activation('sigmoid')(psi_f)
    att_x = layers.multiply([x, rate])
    return att_x

import tensorflow as tf
from tensorflow.keras import layers

# Helper to repeat across channels
#def repeat_elem(tensor, rep):
#    return tf.repeat(tensor, rep, axis=3)

# Attention Gate Block
def attention_block1(x, gating, inter_shape):
    # Get static and dynamic shapes
    static_shape_x = x.shape
    static_shape_g = gating.shape
    shape_x = tf.shape(x)
    shape_g = tf.shape(gating)

    # Step 1: Transform input feature map
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta = tf.shape(theta_x)

    # Step 2: Transform gating signal
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)

    # Compute safe strides
    stride_h = tf.math.maximum(1, shape_theta[1] // shape_g[1])
    stride_w = tf.math.maximum(1, shape_theta[2] // shape_g[2])

    # Step 3: Upsample gating
    upsample_g = layers.Conv2DTranspose(
        inter_shape, (3, 3), strides=(stride_h, stride_w), padding='same'
    )(phi_g)

    # Step 4: Match shapes before adding
    # Adjust theta_x size if needed
    theta_x_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(theta_x)
    
    # Step 5: Combine and apply attention
    concat_xg = layers.add([upsample_g, theta_x_upsampled])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)

    # Step 6: Upsample attention map to x's resolution
    upsample_psi = layers.UpSampling2D(
        size=(static_shape_x[1] // sigmoid_xg.shape[1], static_shape_x[2] // sigmoid_xg.shape[2])
    )(sigmoid_xg)

    # Step 7: Match channel dimension
    upsample_psi = layers.Lambda(lambda z: tf.repeat(z, static_shape_x[3], axis=3))(upsample_psi)

    # Step 8: Apply attention
    y = layers.multiply([upsample_psi, x])

    # Step 9: Output conv + norm
    result = layers.Conv2D(static_shape_x[3], (1, 1), padding='same')(y)
    result = layers.BatchNormalization()(result)

    return result

def attention_block(x, gating, inter_shape):
    # Get static and dynamic shapes
    shape_x = x.shape
    shape_g = gating.shape

    # Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    theta_x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(theta_x)
    shape_theta_x = theta_x.shape

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    def safe_stride(a, b):
        return max(1, a // b)

    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides = (
                                     safe_stride(int(shape_theta_x[1]), int(shape_g[1])),
                                     safe_stride(int(shape_theta_x[2]), int(shape_g[2]))),
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

def attention_up_and_concat(x, skip, pool_size = (2,2)):
    u = MaxUnpooling2D(pool_size)(x)
    c = layers.Concatenate(axis=3)([u, skip])
    return c

def Attention_SegNet(input_shape, n_class=1, pool_size=(2, 2)):
    f = [32, 64, 128, 256, 512, 1024]
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = conv_bn_relu(inputs, f[0])
    c1 = conv_bn_relu(c1, f[0])
    p1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(c1)
    c2 = conv_bn_relu(p1, f[1])
    c2 = conv_bn_relu(c2, f[1])
    p2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(c2)
    c3 = conv_bn_relu(p2, f[2])
    c3 = conv_bn_relu(c3, f[2])
    c3 = conv_bn_relu(c3, f[2])
    p3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(c3)
    c4 = conv_bn_relu(p3, f[3])
    c4 = conv_bn_relu(c4, f[3])
    c4 = conv_bn_relu(c4, f[3])
    p4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(c4)
    c5 = conv_bn_relu(p4, f[4])
    c5 = conv_bn_relu(c5, f[4])
    c5 = conv_bn_relu(c5, f[4])
    p5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(c5)
    
    # Decoder
    def unpool_gate_conv2(x, c, mask, out_filters):
        x = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = MaxUnpooling2D(pool_size)([x, mask])
        g = gating_signal(x, out_filters)
        att = attention_block(c, g, out_filters)
        c = layers.Concatenate(axis=3)([c, att])
        c = conv_bn_relu(c, out_filters)
        c = conv_bn_relu(c, out_filters)
        return c
    
    d1 = unpool_gate_conv2(p5, c5, mask_5, f[4])
    d1 = conv_bn_relu(d1, f[4])
    d2 = unpool_gate_conv2(d1, c4, mask_4, f[3])
    d2 = conv_bn_relu(d2, f[3])
    d3 = unpool_gate_conv2(d2, c3, mask_3, f[2])
    d3 = conv_bn_relu(d3, f[2])
    d4 = unpool_gate_conv2(d3, c2, mask_2, f[1])
    d5 = unpool_gate_conv2(d4, c1, mask_1, f[0])
    
    outputs = layers.Conv2D(n_class, 1, activation='sigmoid' if n_class == 1 else 'softmax')(d5)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy' if n_class == 1 else 'categorical_crossentropy',
                  metrics=['accuracy', jacard_coef, dice_coef])
    
    return model

def Attention_ResSegNet(input_shape, n_class=1, pool_size=(2, 2)):
    f = [32, 64, 128, 256, 512, 1024]
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = residual2_block(inputs, f[0])
    p1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(c1)
    c2 = residual2_block(p1, f[1])
    p2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(c2)
    c3 = residual2_block(p2, f[2])
    p3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(c3)
    c4 = residual2_block(p3, f[3])
    p4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(c4)
    c5 = residual2_block(p4, f[4])
    p5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(c5)
    
    # Decoder
    def unpool_gate(x, c, mask, out_filters):
        x = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = MaxUnpooling2D(pool_size)([x, mask])
        g = gating_signal(x, out_filters)
        att = attention_block(c, g, out_filters)
        c = layers.Concatenate(axis=3)([c, att])
        return c
    
    d1 = unpool_gate(p5, mask_5, f[4])
    d1 = residual2_block(d1, f[4])

    d2 = unpool_gate(d1, mask_4, f[3])
    d2 = residual2_block(d2, f[3])

    d3 = unpool_gate(d2, mask_3, f[2])
    d3 = residual2_block(d3, f[2])

    d4 = unpool_gate(d3, mask_2, f[1])
    d4 = residual2_block(d4, f[1])

    d5 = unpool_gate(d4, mask_1, f[0])
    d5 = residual2_block(d5, f[0])

    outputs = layers.Conv2D(n_class, 1, activation='sigmoid' if n_class == 1 else 'softmax')(d5)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', jacard_coef, dice_coef])
    
    return model
