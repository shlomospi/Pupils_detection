
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
from prep_data import create_ir_data, create_RGB_data, create_RGB2_data
import argparse
from clearml import Task
from tkinter import Tk, filedialog

# ----------------------------------------clear ml init-----------------------------------------------------------------

task = Task.init()

# ----------------------------------------tkinter init-----------------------------------------------------------------

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
# ----------------------------------------GPU check---------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# ----------------------------------------parsing-----------------------------------------------------------------------
def parse_args():

    parser = argparse.ArgumentParser(description="Pupil center detection trainer")
    parser.add_argument("-p", '--phase', type=str, default='train',
                        help='train / retrain')
    parser.add_argument("-e", '--epoch', type=int, default=400,
                        help='The number of epochs to run')
    parser.add_argument("-b", '--batch_size', type=int, default=64,
                        help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--log_dir', type=str, default='',
                        help='Directory name to save training logs')
    parser.add_argument('--blocks', default = 2,
                        help="num of blocks")
    parser.add_argument('--arch', default="arch",
                        help="small/blocks")
    parser.add_argument('-bin', '--binary', default=False, type=bool,
                        help="True for converting data to binary pixels, Flase ")
    parser.add_argument('--data', default="RGB",
                        help="what dataset to load (IR/RGB/Both)")
    parser.add_argument('--threshold', nargs='+', default=(79 // 2, 284 // 2, 0, 255, 0, 107),
                        help=' threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax)')
    return check_args(parser.parse_args())


def check_args(args):
    # --data
    try:
        assert args.data == "RGB" or "IR" or "Both" or "RGB2"
    except ValueError:
        print('Not valid data source')

    # --arch
    try:
        assert args.arch == "blocks" or "small" or "medium"
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
        assert args.phase == "train" or "retrain"
    except ValueError:
        print('phase args must be equal "train", "retrain" or "evaluate"')
    if args.phase == "retrain":
        print("Retraining a model, overwriting Architecture choice")
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
    print("\n\nInit..")
    print("-----------------------------------Parsing-----------------------------------")
    args = parse_args()
    epochs = args.epoch
    learning_rate = args.lr
    batch_size = args.batch_size
    loss = 'mean_squared_error'
    tresh = args.threshold
    # ---------------------log folder creation---------------------------------------------------------------
    print("-----------------------------Creating Log Folder-----------------------------")

    template = "BatchSize_{}_Epochs_{}_LearningRate_{}_model_{}_data_{}_phase_{}"
    config = template.format(batch_size, epochs, str(learning_rate)[2:], args.arch, args.data, args.phase)
    if type(args.log_dir) is str and args.log_dir != "":
        log_folder = "log/{}".format(args.log_dir)
    else:
        date = datetime.now().strftime("%d%m%Y_%H%M")
        log_folder = "log/{}".format(date + "_" + config)
        print("log folder: {}".format(log_folder))
    utils.check_folder(log_folder)  # check and create

    # ----------------------Loading  dataset--------------------------------------------------------------
    print("-------------------------------Loading Dataset-------------------------------")
    binary_message = "converting to binary pixels.." if args.binary else "converting to grayscale pixels.."
    if args.data == "IR": # TODO convert to args.data as a list of sub datasets
        print("Loading IR data, and " + binary_message)
        images, labels = create_ir_data(tresh=tresh, binary=args.binary)

    elif args.data == "RGB":
        print("Loading RGB data, and " + binary_message)
        images, labels = create_RGB_data(tresh=tresh, binary=args.binary, verbose=2)

    elif args.data == "RGB2":
        print("Loading RGB data, and " + binary_message)
        images, labels = create_RGB2_data(tresh=tresh, binary=args.binary, verbose=2)

    elif args.data == "Both":
        print("Loading Both IR and RGB data, and " + binary_message)
        IRimages, IRlabels = create_ir_data(tresh=tresh, binary=args.binary)
        RGBimages, RGBlabels = create_RGB_data(tresh=tresh, binary=args.binary, verbose=2)
        images = np.concatenate((IRimages, RGBimages), axis = 0)
        labels = np.concatenate((IRlabels, RGBlabels), axis = 0)
    elif args.data == "BothRGB":
        print("Loading Both RGB and RGB2 data, and " + binary_message)
        RGB2images, RGB2labels = create_RGB2_data(verbose=2, binary=args.binary)
        RGBimages, RGBlabels = create_RGB_data(verbose=2, binary=args.binary)
        images = np.concatenate((RGB2images, RGBimages), axis=0)
        labels = np.concatenate((RGB2labels, RGBlabels), axis=0)
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
    print("----------------------------Loading & Compiling Model----------------------------------")

    if args.phase == "retrain":
        model_path = filedialog.askdirectory() # Choose
        try:
            model = keras.models.load_model(model_path)
            model.summary()
            model.compile(loss=loss,
                          optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                          metrics=[tf.keras.metrics.MeanSquaredError()])
            image_utils.plot_example_images(x_test, model.predict(x_test),
                                            title="examples from loaded model's predictions before training",
                                            folder=log_folder)
        except AttributeError:
            print("The folder does not contain a saved model.\n Choose log/<training session>/final_model")
        print("Loaded pretrained model from:\n{}".format(model_path))

    else:
        if args.arch == "blocks":
            CNNblocks = args.blocks
            model = models.CNN_regression(input_shape=x_train[4].shape, blocks=CNNblocks)
            print("Loaded Block model with {} blocks".format(CNNblocks))
        elif args.arch == "small":
            model = models.CNN_small_regression(input_shape=x_train[4].shape)
            print("Loaded {} model".format(args.arch))
        elif args.arch == "medium":
            model = models.CNN_medium_regression(input_shape=x_train[4].shape)
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
                                                      verbose=0, mode='auto', cooldown=2, min_lr=learning_rate//100000)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                     verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    callbacks = [checkpointer, earlystopping, adaptivelr]  # selected callbacks

    # -----------------------------------------Train model--------------------------------------------------------------
    print("----------------------------------Training Model---------------------------------------")

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
    print("--------------------------------Evaluating Model---------------------------------------")

    _, train_mse = model.evaluate(x_train, y_train, verbose=0)
    _, validation_mse = model.evaluate(x_val, y_val, verbose=0)
    _, test_mse = model.evaluate(x_test, y_test, verbose=0)
    print('Train MSE: %.5f' % train_mse)
    print('validation MSE: %.5f' % validation_mse)

    print('Test MSE: %.5f' % test_mse)

    utils.plot_acc_lss(history, metric1=loss, metric2=loss, log_dir=log_folder)
    image_utils.plot_example_images(x_test, model.predict(x_test),
                                    title="examples of model's predictions after training",
                                    examples=10, folder=log_folder)

    if epochs > 2:
        print("-------------------------------Logging Experiment--------------------------------------")
        csv_path = os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                'experiment_results.csv')
        print("saveing config and results at: \n{}".format(csv_path))
        arch = "block{}".format(args.blocks) if args.arch == "blocks" else args.arch # TODO test
        # arch = "small" if args.arch == "small" else "block{}".format(args.blocks)
        csv_line = [batch_size, epochs, learning_rate, arch, test_mse, args.data, args.phase]

        with open(csv_path, "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_line)
    print("-------------------------------End off Experiment--------------------------------------")


if __name__ == '__main__':

    main()

    # TODO Augmentation
    # TODO Binary preprocessing

