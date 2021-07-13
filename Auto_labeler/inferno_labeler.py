import argparse
import os
import time

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from image_utils import *
from config import config
from logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO
from utils import annotator, change_channel, gray_normalizer
from tqdm import tqdm
tf.disable_v2_behavior()


def load_model(session, m_type, m_name, logger):
    # load the weights based on best loss
    best_dir = "best_loss"

    # check model dir
    model_path = "models/" + m_name
    path = os.path.join(model_path, best_dir)
    if not os.path.exists(path):
        raise FileNotFoundError

    if m_type == "simple":
        model = Simple(m_name, config, logger)
    elif m_type == "YOLO":
        model = YOLO(m_name, config, logger)
    elif m_type == "GAP":
        model = GAP(m_name, config, logger)
    elif m_type == "NAS":
        model = NASNET(m_name, config, logger)
    elif m_type == "INC":
        model = Inception(m_name, config, logger)
    else:
        raise ValueError

    # load the best saved weights
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        raise ValueError('There is no best model with given model')

    return model


def rescale(image):
    """
    If the input video is other than network size, it will resize the input video
    :param image: a frame form input video
    :return: scaled down frame
    """
    scale_side = max(image.shape)
    # image width and height are equal to 192
    scale_value = config["input_width"] / scale_side

    # scale down or up the input image
    scaled_image = cv2.resize(image, dsize=None, fx=scale_value, fy=scale_value)

    # convert to numpy array
    scaled_image = np.asarray(scaled_image, dtype=np.uint8)

    # one of pad should be zero
    w_pad = int((config["input_width"] - scaled_image.shape[1]) / 2)
    h_pad = int((config["input_width"] - scaled_image.shape[0]) / 2)

    # create a new image with size of: (config["image_width"], config["image_height"])
    new_image = np.ones((config["input_width"], config["input_height"]), dtype=np.uint8) * 250

    # put the scaled image in the middle of new image
    new_image[h_pad:h_pad + scaled_image.shape[0], w_pad:w_pad + scaled_image.shape[1]] = scaled_image

    return new_image


def upscale_preds(_preds, _shapes):
    """
    Get the predictions and upscale them to original size of video
    :param _preds:
    :param _shapes: (y, x)
    :return: upscales x and y
    """
    # we need to calculate the pads to remove them from predicted labels
    pad_side = np.max(_shapes)
    # image width and height are equal to 384
    downscale_value = config["input_width"] / pad_side

    scaled_height = _shapes[0] * downscale_value
    scaled_width = _shapes[1] * downscale_value

    # one of pad should be zero
    w_pad = (config["input_width"] - scaled_width) / 2
    h_pad = (config["input_width"] - scaled_height) / 2

    # remove the pas from predicted label
    x = _preds[0] - w_pad
    y = _preds[1] - h_pad
    w = _preds[2]

    # calculate the upscale value
    upscale_value = pad_side / config["input_height"]

    # upscale preds
    x = x * upscale_value
    y = y * upscale_value
    w = w * upscale_value

    return x, y, w


def videos_to_img(video_folder="videos1", verbose=0):
    """
    create images from frames of videos in folder
    :return: img folder path
    """
    images_folder = "images_from_"+video_folder
    os.makedirs(images_folder)
    counter = 0
    videos = os.listdir(video_folder)
    print("creating images from {} at \n{}".format(video_folder, images_folder))
    for video in videos:
        counter = video_to_img(os.path.join(video_folder, video),
                               images_folder, counter=counter, verbose=verbose)
    return images_folder


def video_to_img(video_path=None, img_path="images_from_video", counter=0, verbose=0):
    """
    takes video as imput and creates a folder of frames from it
    :param verbose:
    :param counter: starts numebring images from counter +1
    :param video_path:
    :param img_path:
    :param freq:
    :return: last frame #
    """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    img_folder_path = os.path.join(fileDir, img_path)
    print("proccessing ", video_path)
    if verbose > 3:
        print("starting from image number {}".format(counter + 1))
    cap = cv2.VideoCapture(video_path)
    if verbose > 1:
        cv.namedWindow("input", cv.WINDOW_NORMAL)
        cv.resizeWindow('input', 256, 144)
    ret = True
    counter = counter
    while ret:
        ret, frame = cap.read()

        if ret:

            if verbose > 1:
                cv.imshow('input', frame)
            counter += 1
            image_name = "{}.jpg".format(counter)
            image_path = os.path.join(img_folder_path, image_name)
            img = cv.resize(frame, (256, 144), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            cv2.imwrite(image_path, img)
            if cv.waitKey(1) == ord('q'):
                break
    return counter


# load a the model with the best saved state from file and predict the pupil location
# on the input video. finaly save the video with the predicted pupil on disk
def main(m_type, m_name, logger, video_path=None, write_output=True):
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)

        # check input source is a file or camera
        if video_path == None:
            video_path = 0

        # load the video or camera
        cap = cv2.VideoCapture(video_path)
        ret = True
        counter = 0
        tic = time.time()
        frames = []
        preds = []

        while ret:
            ret, frame = cap.read()

            if ret:
                # Our operations on the frame come here
                frames.append(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                f_shape = frame.shape
                if frame.shape[0] != 192:
                    frame = rescale(frame)

                image = gray_normalizer(frame)
                image = change_channel(image, config["input_channel"])
                [p] = model.predict(sess, [image])
                x, y, w = upscale_preds(p, f_shape)

                preds.append([x, y, w])
                # frames.append(gray)
                counter += 1

        toc = time.time()
        print("{0:0.2f} FPS".format(counter / (toc - tic)))

    # get the video size
    video_size = frames[0].shape[0:2]
    if write_output:
        # prepare a video write to show the result
        video = cv2.VideoWriter("predicted_video.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                (video_size[1], video_size[0]))

        for i, img in enumerate(frames):
            labeled_img = annotator((0, 250, 0), img, *preds[i])
            video.write(labeled_img)

        # close the video
        cv2.destroyAllWindows()
        video.release()
    print("Done...")


def main_images(m_type, m_name, logger, folder_path=None, write_output=True, verbose=2):
    """
    takes a folder of images, and creates a dataset in a folder with
    numbered jpg images and coresponding line in a csv with the landmark
    :param m_type:
    :param m_name:
    :param logger:
    :param folder_path:
    :param write_output:
    :param verbose:
    :return:
    """
    with tf.Session() as sess:

        fileDir = os.path.dirname(os.path.realpath('__file__'))

        outPath = os.path.join(fileDir,  f"{folder_path}_dataset")
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        path = os.path.join(fileDir, folder_path)

        frames = []
        preds = []

        # load best model
        model = load_model(sess, m_type, m_name, logger)
        # iterate through the names of contents of the folder
        img_num = 0
        cv.namedWindow("output", cv.WINDOW_NORMAL)
        cv.resizeWindow('output', 900, 600)
        for image_name in tqdm(os.listdir(path)):
            img_num += 1
            # create the full input path and read the file
            input_path = os.path.join(path, image_name)
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)

            frames.append(image)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f_shape = frame.shape # y, x
            if frame.shape[0] != 192:
                frame = rescale(frame)

            proccessed_image = gray_normalizer(frame)
            proccessed_image = change_channel(proccessed_image, config["input_channel"])
            [p] = model.predict(sess, [proccessed_image])

            x, y, w = upscale_preds(p, f_shape)
            preds.append([x, y, w])
            if verbose > 2:
                cv.imshow("labeled", cross_annotator(image, (x, y), size=10))
                if cv.waitKey(1) == ord('q'):
                    break
            with open(outPath+'/labels.txt', "a") as txtfile:
                txtfile.write(str(round(x, 2)) + " " + str(round(y, 2)) + "\n")

            cv2.imwrite(outPath+'/{}.jpg'.format(img_num), image)

        if write_output:
            # prepare a video write to show the result

            video = cv2.VideoWriter("predicted_ds_video.avi", cv2.VideoWriter_fourcc(*"XVID"), 2, (192, 192))

            for i, img in enumerate(frames):
                labeled_img = annotator((0, 250, 0), img, *preds[i])
                video.write(np.asarray(labeled_img))

            # close the video
            cv2.destroyAllWindows()
            video.release()
    print("Done...")


if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)

    parser.add_argument('--model_type',
                        help="INC, YOLO, simple",
                        default="INC")

    parser.add_argument('--model_name',
                        help="name of saved model (3A4Bh-Ref25)",
                        default="3A4Bh-Ref25")

    parser.add_argument('--videos_folder',
                        help="path to video folder",
                        default=None)

    parser.add_argument('--image_path',
                        help="path to folder of images",
                        default=None)

    args = parser.parse_args()

    # model_name = args.model_name
    model_name = args.model_name
    model_type = args.model_type
    videos_folder = args.videos_folder
    folder_path = args.image_path

    # initial a logger
    logger = Logger(model_type, model_name, "", config, dir="models/")

    if videos_folder is not None:
        if str(videos_folder) == "0":
            img_folder = "images_from_camera"
            os.makedirs(img_folder)
            path_to_images = img_folder
            video_to_img(0, img_path=img_folder, verbose=2)
        else:
            path_to_images = videos_to_img(videos_folder)
        main_images(model_type, model_name, logger, path_to_images)
    elif folder_path is not None:
        logger.log("image folder mode")
        main_images(model_type, model_name, logger, folder_path, verbose=0)
