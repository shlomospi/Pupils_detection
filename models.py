import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Flatten, concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, ReLU, AvgPool2D
from tensorflow.keras.regularizers import l2


def vgg_block(input_vec, filters=32, l2_weight_regulaizer=0.0002, weight_initializer="he_uniform", kernel=(3, 3)):
    """
    two conv layers with batchnorm afterwards and maxpooling in the end
    :param input_vec:
    :param filters:
    :param l2_weight_regulaizer:
    :param weight_initializer:
    :param kernel:
    :return:
    """

    with tf.keras.backend.name_scope("VGG_Block"):
        conv3x3_1 = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                           kernel_regularizer=l2(l2_weight_regulaizer))(input_vec)
        batch_norm_1 = BatchNormalization()(conv3x3_1)
        conv3x3_2 = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                           kernel_regularizer=l2(l2_weight_regulaizer))(batch_norm_1)
        batch_norm_2 = BatchNormalization()(conv3x3_2)

        return MaxPooling2D(pool_size=2)(batch_norm_2)


def vgg_model(input_shape, l2_weight_regulaizer=0.0002, weight_initializer="he_uniform", num_classes=10):

    print("loading vgg_model as model..")
    inputs = tf.keras.layers.Input(input_shape)

    vgg_blk1 = vgg_block(inputs, filters=32,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)

    vgg_blk2 = vgg_block(vgg_blk1, filters=64,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)
    vgg_blk3 = vgg_block(vgg_blk2, filters=128,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)

    vgg_blk4 = vgg_block(vgg_blk3, filters=256,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)
    vgg_blk5 = vgg_block(vgg_blk4, filters=512, l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)
    flatten_1 = Flatten()(vgg_blk5)
    dense_1 = Dense(512, activation='relu', kernel_initializer=weight_initializer,
                    kernel_regularizer=l2(0.0001))(flatten_1)
    drop_1 = Dropout(0.5)(dense_1)
    """dense_2 = Dense(128, activation='relu', kernel_initializer=weight_initializer,
                    kernel_regularizer=l2(0.0001))(drop_1)
    drop_2 = Dropout(0.5)(dense_2)"""
    outputs = Dense(num_classes)(drop_1) #, activation='sigmoid')(drop_1)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def CNN_regression(input_shape, filters=32, l2_weight_regulaizer=0.0002, weight_initializer="he_uniform", kernel=(3, 3), blocks=4):
    """
    two conv layers with batchnorm afterwards and maxpooling in the end
    :param input_shape:
    :param filters:
    :param l2_weight_regulaizer:
    :param weight_initializer:
    :param kernel:
    :return:
    """
    inputs = tf.keras.layers.Input(input_shape)
    print("Building CNN regression model")
    for i in range(blocks):
        if i == 0:

            x = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                       kernel_regularizer=l2(l2_weight_regulaizer))(inputs)
        else:
            x = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                       kernel_regularizer=l2(l2_weight_regulaizer))(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                   kernel_regularizer=l2(l2_weight_regulaizer))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(8)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model
