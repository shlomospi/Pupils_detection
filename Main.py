
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np

import image_utils
import models
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import utils
from prep_data import create_ir_data
import argparse
from clearml import Task

# ----------------------------------------clear ml init-----------------------------------------------------------------

task = Task.init()
# ----------------------------------------GPU check---------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# ----------------------------------------parsing-----------------------------------------------------------------------
def parse_args():

    parser = argparse.ArgumentParser(description="Tensorflow implementation of ModuleNet")
    parser.add_argument("-p", '--phase', type=str, default='train', help='train / image / video') # TODO add option to predict img and vid
    parser.add_argument("-e", '--epoch', type=int, default=400, help='The number of epochs to run')
    parser.add_argument("-b", '--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--log_dir', type=str, default='', help='Directory name to save training logs')
    parser.add_argument('--blocks', type=int, default = 3, help="size of network")
    return check_args(parser.parse_args())


def check_args(args):

    # --blocks
    try:
        assert type(args.blocks) is int
        assert args.blocks >= 0
    except ValueError:
        print('#blocks must be larger than zero')
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
    print("\n\n Init....")
    args = parse_args()
    CNNblocks = args.blocks
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
    loss = 'mean_squared_error'
    # ---------------------log folder creation---------------------------------------------------------------
    if type(args.log_dir) is str and args.log_dir != "":
        log_folder = "log/{}".format(args.log_dir)
    else:
        config = "BatchSize_{}_Epochs_{}_LearningRate_{}".format(batch_size, epochs, str(learning_rate)[2:])
        date = datetime.now().strftime("%d%m%Y_%H%M")
        log_folder = "log/{}".format(date + "_" + config)
        print("log folder: {}".format(log_folder))
    utils.check_folder(log_folder)  # check and create

    # ----------------------Loading  dataset--------------------------------------------------------------
    print("Loading IR data")
    images, labels = create_ir_data()
    if images.ndim == 3: # for 1 channel images
        images = np.expand_dims(images, axis=-1)
    print("Split")
    x_train, x_test, y_train, y_test = train_test_split(images, labels , test_size=0.1, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=7)
    print('Train data size: {}, train label size: {}'.format(x_train.shape, y_train.shape))
    print('val data size: {}, val label size: {}'.format(x_val.shape, y_val.shape))
    print('test data size: {}, test label size: {}'.format(x_test.shape, y_test.shape))

    # -------------------------------------Load & compile model---------------------------------------------------------

    model = models.CNN_regression(input_shape= x_train[4].shape, blocks=CNNblocks)
    model.compile(loss=loss,
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    # -------------------------------------Callbacks--------------------------------------------------------------------

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=log_folder, verbose=1,
                                                      monitor='val_loss', save_best_only=True)

    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_folder, histogram_freq=1,
    #                                              write_images=True, write_graph=False)

    adaptivelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=4,
                                                      verbose=0, mode='auto', cooldown=2, min_lr=0.00001)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                     verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    callbacks = [checkpointer, earlystopping, adaptivelr]  # selected callbacks

    # -----------------------------------------Train model--------------------------------------------------------------
    if train:
        print(f"\n\nTraining starting with batches of {batch_size}, LR of {learning_rate} for {epochs} epochs.")
        history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            verbose=2,
                            callbacks=callbacks)

        print(f"Saving best result at {config}_saved_model")
        model.save(config+'_saved_model')

    # ----------------------------------Load "saved_model"-------------------------------------------------------------
    else:
        print("Loading saved_model")
        model = keras.models.load_model('saved_model')

    #  ---------------------------------------Evaluate the model ------------------------------------------------------

    if train:
        _, train_mse = model.evaluate(x_train, y_train, verbose=0)
        _, validation_mse = model.evaluate(x_val, y_val, verbose=0)
        print('Train MSE: %.3f' % train_mse)
        print('validation MSE: %.3f' % validation_mse)
        _, test_mse = model.evaluate(x_test, y_test, verbose=0)
        print('Test MSE: %.3f' % test_mse)

        utils.plot_acc_lss(history, metric1=loss, metric2=loss, log_dir=log_folder)
        # TODO add csv file experiment logger (args, loss)
        image_utils.plot_example_images(x_test, model.predict(x_test), examples=10, folder=log_folder)


if __name__ == '__main__':

    main()
