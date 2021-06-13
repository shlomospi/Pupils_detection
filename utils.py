import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import os
import cv2 as cv


def down_scale(image, target_x=30, target_y=50):
    """
    If the input video is other than network size, it will resize the input video
    :param image: a frame form input video
    :return: scaled down frame
    """
    original_x, original_y = image.shape
    #
    scale_x = target_x / original_x
    scale_y = target_y / original_y

    # scale down or up the input image
    scaled_image = cv.resize(image, dsize=(target_x, target_y), interpolation = cv.INTER_CUBIC)

    # convert to numpy array
    scaled_image = np.asarray(scaled_image, dtype=np.uint8)

    return scaled_image, scale_x, scale_y


def plot_acc_lss(history_log, log_dir=None, metric1='categorical_accuracy', metric2='loss',verbose=1):
    """
    plot loss and acc via plt
    Args:
        history_log: history file of training session
        log_dir: dir to save the plot
        verbose: 1 for showing the plot 0 for not showing the plot

    Returns: NaN
    :param verbose:
    :param metric2:
    :param history_log:
    :param log_dir:
    :param metric1:

    """
    plt.figure(figsize=(8, 6))
    # ----summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_log.history[metric1])
    plt.plot(history_log.history["val_"+metric1])
    plt.title(metric1)
    plt.ylabel(metric1)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(b=True, which='Both', axis='both')
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history_log.history[metric2])
    plt.plot(history_log.history['val_'+metric2])
    plt.title(metric2)
    plt.ylabel(metric2)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(b=True, which='Both', axis='both')

    if log_dir is None:
        while True:
            given_dir = input("To save training plot, provide dir location. leave empty to skip")
            if given_dir == "":
                break
            else:
                try:
                    plt.savefig('{}/training_plots.png'.format(given_dir))
                except ValueError:
                    print("could not save in provided dir: {}".format(given_dir))
                    continue
                else:
                    break
    else:

        plt.savefig('{}/training_plots.png'.format(log_dir))

    if verbose >= 1:
        plt.show()


def horizontal_flip_and_show(data, labels, verbose=0):
    """
    tf.image.flip_left_right wrapper. adds left right augmented images and labels
    :param data: image tensor
    :param labels: one hot label tensor
    :param verbose: 1 to show random example
    :return: data with flipped addition, new labels
    """
    with tf.device('/device:cpu:0'):
        fliped_data = tf.image.flip_left_right(data)
        if verbose >= 1:

            rand = random.randint(0, 1000)
            plt.subplot(1, 2, 1)
            plt.title('Original Image #{}'.format(rand))
            plt.imshow(data[rand, :, :, :])
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title('Augmented Image #{}'.format(rand))
            plt.imshow(fliped_data[rand, :, :, :])
            plt.axis('off')
            plt.show()

    return tf.concat([data, fliped_data], 0), tf.concat([labels, labels], 0)


def check_folder(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
