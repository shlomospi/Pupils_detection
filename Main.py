import tensorflow as tf
import numpy as np
import csv
import image_utils
import models
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import utils
from prep_data import *
import argparse

from tkinter import Tk, filedialog
from config import config

# ----------------------------------------tkinter init-----------------------------------------------------------------

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
# ----------------------------------------GPU check---------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# ----------------------------------------parsing-----------------------------------------------------------------------
def parse_args():

    parser = argparse.ArgumentParser(description="fast Pupil center detection trainer")
    parser.add_argument("-p", '--phase',
                        type=str,
                        default='train',
                        help='train / retrain. \n'
                             'retraining will ask for a "saved_model" file and '
                             'ignore any architecture choices')

    parser.add_argument("-e", '--epoch',
                        type=int,
                        default=400,
                        help='The maximal number of epochs to run')
    parser.add_argument("-d", '--data',
                        nargs='+',
                        default=["RGB"],
                        help='what data to load. available: RGB RGB2 IR')
    parser.add_argument("-b", '--batch_size',
                        type=int,
                        default=64,
                        help='The size of a batch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('-rlr', '--reducelr',
                        type=int,
                        default=100000,
                        help='by how much to reduce learning rate')
    parser.add_argument('-log', '--log_dir',
                        type=str,
                        default='',
                        help='Directory name to save training logs, '
                             'any pictures and saved models')
    parser.add_argument('-bk', '--blocks',
                        default = 2,
                        help="num of blocks for a block type architecture")
    parser.add_argument('--arch',
                        default="medium",
                        help="small/blocks/medium")
    parser.add_argument('-bin', '--binary',
                        default=False,
                        type=bool,
                        help="'-bin True' for converting data to binary pixels, "
                             "ignore for False ")
    parser.add_argument('-t', '--threshold',
                        nargs='+',
                        default=["0"],
                        help=' threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax) for '
                             'image preproccessing. or an int for picked values for dictionary')
    parser.add_argument('--filters',
                        nargs='+',
                        default=(32, 64, 128),
                        help='filters for "medium" net')
    parser.add_argument('-a', '--augmentation',
                        nargs='+',
                        default=["imgaug"],
                        help='what augmentation to do? "imgaug"')

    return check_args(parser.parse_args())


def check_args(args):

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
        print('phase args must be equal "train" or "retrain"')
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


def main(verbose = 0):
    #  -----------------------------------------parameters------------------------------------------------------------
    print("\n\nInit..")
    print("-----------------------------------Parsing-----------------------------------")
    args = parse_args()
    epochs = args.epoch
    learning_rate = args.lr
    batch_size = args.batch_size
    loss = 'mean_squared_error'
    Thresholds = config["thresholds"]
    resolution = config["res"]
    if len(args.threshold) == 6:
        threshold = args.threshold
        threshold_log = "_".join(args.threshold)
    else:
        assert len(args.threshold) == 1

        threshold = Thresholds[args.threshold[0]]
        threshold_log = args.threshold[0]

    # ---------------------log folder creation---------------------------------------------------------------
    print("-----------------------------Creating Log Folder-----------------------------")

    pixel_format = "binary" if args.binary else "grayscale"

    aug_log = ""
    for augmentation in args.augmentation:
        aug_log += str(augmentation) + "_"
    if len(aug_log) > 1:
        aug_log = aug_log[:-1]

    if args.arch == "medium":
        arch = "medium"
        for f in args.filters:
            arch += "_{}".format(f)

    elif args.arch == "blocks":
        arch = "block{}".format(args.blocks)
    else:
        arch = args.arch

    template = "BatchSize_{}_Epochs_{}_LearningRate_{}_model_{}_data_{}_phase_{}_aug_{}_thresh_{}"
    if args.binary:
        template += "_binary"

    config_log = template.format(batch_size,
                                 epochs,
                                 str(learning_rate)[2:],
                                 arch,
                                 "".join(args.data),
                                 args.phase,
                                 aug_log,
                                 threshold_log)
    main_log_folder = "log_" + "_".join([str(coord) for coord in config["res"]])
    if type(args.log_dir) is str and args.log_dir != "":
        log_folder = os.path.join(main_log_folder, args.log_dir) # "log/{}".format(args.log_dir)
    else:
        date = datetime.now().strftime("%d%m%Y_%H%M")
        log_folder = os.path.join(main_log_folder, date + "_" + config_log) # "log/{}".format(date + "_" + config_log)
        print("log folder: {}".format(log_folder))
    utils.check_folder(log_folder)  # check and create
    # TODO create config file in log folder, load from it at inference
    # ----------------------Loading  dataset--------------------------------------------------------------
    print("-------------------------------Loading Dataset-------------------------------")
    images, labels = prep_data(data=args.data,
                               binary=args.binary,
                               res=resolution,
                               thresh=threshold,
                               verbose = verbose,
                               normalize=False)
    print("Spliting..")
    x_train, x_test, y_train, y_test = train_test_split(images,
                                                        labels,
                                                        test_size=0.1,
                                                        random_state=42)

    if x_test.ndim == 3:  # for 1 channel images
        x_test = np.expand_dims(images, axis=-1)

    print("augmenting..")
    if "imgaug" in augmentation:
        ne_images, ne_labels = augmentor(images,
                                         labels,
                                         someof=2,
                                         augments_num=20,
                                         verbose=2)
        x_train, y_train = concat_datasets(x_train, y_train, ne_images, ne_labels)

    y_train = normalize_coord_tensor(y_train, resolution)
    y_test = normalize_coord_tensor(y_test, resolution)

    print("splitting valset from testset")
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=7)
    print("final datasets sizes:")
    print('Train data size: {}, train label size: {}'.format(x_train.shape, y_train.shape))
    print('val data size:   {}, val label size:   {}'.format(x_val.shape, y_val.shape))
    print('test data size:  {}, test label size:  {}'.format(x_test.shape, y_test.shape))
    if verbose > 2:
        image_utils.plot_example_images(x_train, y_train,
                                        title="examples from the training dataset")
        image_utils.plot_example_images(x_val, y_val,
                                        title="examples from the val dataset")
        image_utils.plot_example_images(x_test, y_test,
                                        title="examples from the test dataset")
    # -------------------------------------Load & compile model---------------------------------------------------------
    print("----------------------------Loading & Compiling Model----------------------------------")

    if args.phase == "retrain":

        model_path = filedialog.askdirectory() # prompt user to choose a folder from which a saved model will be loaded
        try:
            model = tf.keras.models.load_model(model_path)
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
            l2_weight_regulaizer = 0.0008
            model = models.CNN_medium_regression(input_shape=x_train[4].shape,
                                                 filters=args.filters,
                                                 l2_weight_regulaizer = l2_weight_regulaizer)
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
                                                      verbose=0, mode='auto', cooldown=2, min_lr=learning_rate//args.reducelr)

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

    #  ---------------------------------------Evaluate the model ------------------------------------------------------
    print("--------------------------------Evaluating Model---------------------------------------")

    _, train_mse = model.evaluate(x_train, y_train, verbose=0)
    _, validation_mse = model.evaluate(x_val, y_val, verbose=0)
    _, test_mse = model.evaluate(x_test, y_test, verbose=0)
    print('Train MSE: %.7f' % train_mse)
    print('validation MSE: %.7f' % validation_mse)

    print('Test MSE: %.7f' % test_mse)

    utils.plot_acc_lss(history, metric1=loss, metric2=loss, log_dir=log_folder)
    image_utils.plot_example_images(x_test, model.predict(x_test),
                                    title="examples of model's predictions after training on testset",
                                    examples=10, folder=log_folder)
    if verbose > 2:
        image_utils.plot_example_images(x_val, model.predict(x_val),
                                        title="examples of model's predictions after training on valset",
                                        examples=10, folder=log_folder)
        image_utils.plot_example_images(x_train, model.predict(x_train),
                                        title="examples of model's predictions after training on trainingset",
                                        examples=10, folder=log_folder)
    if epochs > 2:
        print("-------------------------------Logging Experiment--------------------------------------")
        csv_path = os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                'experiment_results.csv')
        print("saving config and results at: \n{}".format(csv_path))

        csv_line = [batch_size, epochs, learning_rate, arch, test_mse, args.data, args.phase,
                    pixel_format, aug_log, threshold_log]

        with open(csv_path, "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_line)
    print("-------------------------------End off Experiment--------------------------------------")


if __name__ == '__main__':

    main()
