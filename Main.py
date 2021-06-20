
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import csv
import image_utils
import models
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import utils
from prep_data import create_ir_data, create_RGB_data
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
    parser.add_argument("-p", '--phase', type=str, default='train', help='train / retrain / evaluate')
    parser.add_argument("-e", '--epoch', type=int, default=400, help='The number of epochs to run')
    parser.add_argument("-b", '--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--log_dir', type=str, default='', help='Directory name to save training logs')
    parser.add_argument('--blocks', default = 2, help="num of blocks")
    parser.add_argument('--arch', default="arch", help="small/blocks")
    parser.add_argument('--data', default="RGB", help="what dataset to load (IR/RGB/Both)")
    return check_args(parser.parse_args())


def check_args(args):
    # --data
    try:
        assert args.data == "RGB" or "IR" or "Both"
    except ValueError:
        print('Not valid data source')

    # --arch
    try:
        assert args.arch == "blocks" or "small"
    except ValueError:
        print('Not valid architecture type')

    # --blocks
    try:
        assert type(args.blocks) is int
        assert args.blocks > 1
    except ValueError:
        print('Not valid block architecture')

    # --phase
    try:
        assert args.phase == "train" or "evaluate" or "retrain"
    except ValueError:
        print('phase args must be equal "train", "retrain" or "evaluate"')

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

    train = True if args.phase == "train" or "retrain" else False
    if type(args.epoch) is int:
        epochs = args.epoch
    else:
        epochs = 400

    if type(args.lr) is float:
        learning_rate = args.lr
    else:
        learning_rate = 0.0002

    if type(args.batch_size) is int:
        batch_size = args.batch_size
    else:
        batch_size = 64
    loss = 'mean_squared_error'
    # ---------------------log folder creation---------------------------------------------------------------
    template = "BatchSize_{}_Epochs_{}_LearningRate_{}_model_{}_data{}_phase{}"
    config = template.format(batch_size, epochs, str(learning_rate)[2:], str(args.arch), args.data, args.phase)
    if type(args.log_dir) is str and args.log_dir != "":
        log_folder = "log/{}".format(args.log_dir)
    else:
        date = datetime.now().strftime("%d%m%Y_%H%M")
        log_folder = "log/{}".format(date + "_" + config)
        print("log folder: {}".format(log_folder))
    utils.check_folder(log_folder)  # check and create

    # ----------------------Loading  dataset--------------------------------------------------------------
    if args.data == "IR":
        print("Loading IR data..")
        images, labels = create_ir_data()
    elif args.data == "RGB":
        print("Loading RGB data..")
        images, labels = create_RGB_data(verbose=2)

    elif args.data == "Both":
        print("Loading Both IR and RGB data")
        IRimages, IRlabels = create_ir_data()
        RGBimages, RGBlabels = create_RGB_data(verbose=2)
        images = np.concatenate((IRimages, RGBimages), axis = 0)
        labels = np.concatenate((IRlabels, RGBlabels), axis = 0)

    else:
        raise SystemExit("Typo is dataset name")

    if images.ndim == 3: # for 1 channel images
        images = np.expand_dims(images, axis=-1)
    print("Split:")
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=7)
    print('Train data size: {}, train label size: {}'.format(x_train.shape, y_train.shape))
    print('val data size:   {}, val label size:   {}'.format(x_val.shape, y_val.shape))
    print('test data size:  {}, test label size:  {}'.format(x_test.shape, y_test.shape))
    image_utils.plot_example_images(x_train, y_train,
                                    title="examples from the dataset")

    # -------------------------------------Load & compile model---------------------------------------------------------
    if args.phase == "retrain":
        model = keras.models.load_model('saved_model')
        print("Loaded pretrained model")
        model.summary()
        model.compile(loss=loss,
                      optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[tf.keras.metrics.MeanSquaredError()])
        image_utils.plot_example_images(x_test, model.predict(x_test),
                                        title="examples from loaded model's predictions before training",
                                        folder=log_folder)
    else:
        if args.arch == "blocks":
            CNNblocks = args.blocks
            model = models.CNN_regression(input_shape=x_train[4].shape, blocks=CNNblocks)
            print("Loaded Block model with {} blocks".format(CNNblocks))
        elif args.arch == "small":
            model = models.CNN_small_regression(input_shape=x_train[4].shape)
            print("Loaded {} model".format(args.arch))
        else:
            raise SystemExit("no model found", args.arch, type(args.arch))

        model.compile(loss=loss,
                      optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[tf.keras.metrics.MeanSquaredError()])

    # -------------------------------------Callbacks--------------------------------------------------------------------

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=log_folder, verbose=1,
                                                      monitor='val_loss', save_best_only=True)

    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_folder, histogram_freq=1,
    #                                              write_images=True, write_graph=False)

    adaptivelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=3,
                                                      verbose=0, mode='auto', cooldown=2, min_lr=learning_rate//1000)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                     verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    callbacks = [checkpointer, earlystopping, adaptivelr]  # selected callbacks

    # -----------------------------------------Train model--------------------------------------------------------------

    print(f"\n\nTraining starting with batches of {batch_size}, LR of {learning_rate} for {epochs} epochs.")
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        verbose=2,
                        callbacks=callbacks)
    final_model_path = os.path.join(log_folder, "final_model")
    print(f"Saving best result at {final_model_path}")
    model.save(final_model_path)
    """
    # ----------------------------------Load "saved_model"-------------------------------------------------------------
    if not train: # TODO remove
        print("Loading saved_model")
        model = keras.models.load_model('saved_model')
    """
    #  ---------------------------------------Evaluate the model ------------------------------------------------------

    _, train_mse = model.evaluate(x_train, y_train, verbose=0)
    _, validation_mse = model.evaluate(x_val, y_val, verbose=0)
    print('Train MSE: %.3f' % train_mse)
    print('validation MSE: %.3f' % validation_mse)
    _, test_mse = model.evaluate(x_test, y_test, verbose=0)
    print('Test MSE: %.3f' % test_mse)

    utils.plot_acc_lss(history, metric1=loss, metric2=loss, log_dir=log_folder)
    image_utils.plot_example_images(x_test, model.predict(x_test),
                                    title="examples of model's predictions after training",
                                    examples=10, folder=log_folder)

    if epochs > 2:
        csv_path = os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                'experiment_results.csv')
        print("saveing config and results at: \n{}".format(csv_path))
        arch = "small" if args.arch == "small" else "block{}".format(args.blocks)
        csv_line = [batch_size, epochs, learning_rate, arch, test_mse, args.data, args.phase]

        with open(csv_path, "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_line)


if __name__ == '__main__':

    main()

    # TODO search hyperparam

