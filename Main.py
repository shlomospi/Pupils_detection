
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import models
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import utils
from prep_data import create_ir_data
import argparse

# ----------------------------------------GPU check---------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# ----------------------------------------parsing-----------------------------------------------------------------------
def parse_args():

    parser = argparse.ArgumentParser(description="Tensorflow implementation of ModuleNet")
    parser.add_argument("-p", '--phase', type=str, default='train', help='train or evaluate ?')
    parser.add_argument("-e", '--epoch', type=int, default=400, help='The number of epochs to run')
    parser.add_argument("-b", '--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--log_dir', type=str, default='', help='Directory name to save training logs')

    return check_args(parser.parse_args())


def check_args(args):

    # --phase
    try:
        assert args.phase == "train" or "evaluate"
    except ValueError:
        print('phase args must be equal "train" or "evaluate"')

    # --epoch
    try:
        assert args.epoch >= 1
    except ValueError:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except ValueError:
        print('batch size must be larger than or equal to one')

    # --lr
    try:
        assert args.lr >= 0
    except ValueError:
        print('learning rate must be larger than zero')

    return args


def main():
    #  -----------------------------------------parameters------------------------------------------------------------
    args = parse_args()
    special_data = True  # restructure data for challenge
    cheat = True  # use only train and validation sets.
    train = True if args.phase == "train" else False
    if type(args.epoch) is int:
        epochs = args.epoch
    else:
        epochs = 400

    if type(args.lr) is int:
        learning_rate = args.lr
    else:
        learning_rate = 0.0002

    if type(args.batch_size) is int:
        batch_size = args.batch_size
    else:
        batch_size = 64

    # log folder creation
    if type(args.log_dir) is str and args.log_dir != "":
        log_folder = "log/{}".format(args.log_dir)
    else:
        config = "BatchSize_{}_Epochs_{}_LearningRate_{}".format(batch_size, epochs, str(learning_rate)[2:])
        date = datetime.now().strftime("%d%m%Y_%H%M")
        log_folder = "log/{}".format(date + "_" + config)

    utils.check_folder(log_folder)  # check and create

    # ----------------------Loading CIFAR10 dataset--------------------------------------------------------------

    (x_train, y_train), (x_test, y_test) = train_test_split(create_ir_data(), test_size=0.1, random_state=42)# TODO  # load the data
