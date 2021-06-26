import cv2 as cv
import argparse
import os
import numpy as np
from image_utils import *
import utils
from tqdm import tqdm
import tensorflow as tf
# import tensorflow_addons as tfa
import image_utils
from config import config
import imageio
import imgaug as ia
from imgaug import augmenters as iaa


def video_csv_to_np_wrapper(data_path = "",
                            txt_file = "1.txt",
                            video_file = "1.avi",
                            thresh=(79//2, 284//2, 0, 255, 0, 107),
                            res=(64, 32),
                            binary=False):
    """
    wrapper for the video_csv_to_np function, creates full path and asserts files are in path
    :param res:
    :param binary:
    :param thresh:
    :param data_path:q
    :param txt_file:
    :param video_file:
    :return:
    """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    txt_path = os.path.join(fileDir, data_path, txt_file)
    video_path = os.path.join(fileDir, data_path, video_file)
    if not os.path.isfile(txt_path):
        print(txt_path)
        raise FileNotFoundError

    if not os.path.isfile(video_path):
        print(video_path)
        raise FileNotFoundError

    return video_csv_to_np(video_path, txt_path, res=res, thresh=thresh, binary=binary)


def video_csv_to_np(videopath, csvpath, res=(64, 32), augment=False,
                    thresh=(79//2, 284//2, 0, 255, 0, 107), binary=False, verbose=0):
    """
    creates a dataset from a video and a csv file.
    :param videopath: path to video
    :param verbose:
    :param res: final resolution for images
    :param csvpath: path to CSV file
    :param augment: Flase for normalization of landmarks
    :param thresh: threshold limits
    :param binary: True for binary pixels, false for grayscale
    :return: y_dataset, x_dataset
    """
    cap = cv.VideoCapture(videopath)
    original_res = cap.get(3), cap.get(4)
    scale = original_res[0] / res[0], original_res[1] / res[1]
    if verbose >= 3:
        print("Original res: {}".format(original_res))
        print("Resize Scale: {}".format(scale))
        print("New Res :{}".format(res))

    label_list = []
    array_list = []
    with open(csvpath, "r") as f:
        for line in f:
            #  values: [ x, y]
            values = line.strip().split(" ")
            values = list(map(float, values))

            if not augment:
                values = normalize_coord(values, original_res)

            label_list.append([values[0], values[1]])
    label_list = np.asarray(label_list)
    if verbose >= 2:
        print("label shape:        ", np.shape(label_list), type(label_list))

    cap = cv.VideoCapture(videopath)

    while True:

        ret, frame = cap.read()

        if ret:

            array_list.append(img_preproccess(frame, res=res,
                                              thresh=thresh, binary=binary))
        else:
            break

    array_list = np.asarray(array_list)

    if verbose >= 2:
        print("images array shape: ", np.shape(array_list), type(array_list))

    return label_list, array_list


def create_ir_data(thresh=(79 // 2, 284 // 2, 0, 255, 0, 107),
                   res=(64, 32),
                   binary=False,
                   verbose=2):
    """
    creates a dataset from videos and a csv file in the "IR videos" directory.
    This dataset was created from 12 IR videos.
    :param res:
    :param thresh: threshold limits
    :param binary: True for binary pixels, false for grayscale
    :param verbose:
    :return: x_dataset, y_dataset
    """
    label_mat = []
    img_mat = []
    for vid in tqdm([1, 2, 3, 4, 5, 9, 13, 21, 22, 91, 111, 121]):
        txt_file = str(vid)+".txt"
        video_file = str(vid)+".avi"

        labels_, images = video_csv_to_np_wrapper(data_path = "IR videos/",
                                                  txt_file=txt_file,
                                                  video_file=video_file,
                                                  thresh=thresh,
                                                  res=res,
                                                  binary=binary)
        label_mat.append(labels_)
        img_mat.append(images)

    label_mat = np.concatenate(np.asarray(label_mat))
    img_mat = np.concatenate(np.asarray(img_mat))
    if verbose >= 2:
        print("IR dataset:")
        print("images array shape: ", np.shape(img_mat), type(img_mat))
        print("labels array shape: ", np.shape(label_mat), type(label_mat))
    return img_mat, label_mat


def create_RGB_data(res=(64, 32), normalize=True, thresh=(79 // 2, 284 // 2, 0, 255, 0, 107),
                    binary=False, verbose=0):
    """
        creates a dataset from images and a csv file in the "images_dataset" directory.
        This dataset was created by inference of the inferno_labeler.py script on images.
        :param res: x,y final resolution
        :param normalize: Should the landmarks be normalized
        :param thresh: threshold limits
        :param binary: True for binary pixels, False for grayscale
        :param verbose:
        :return: x_dataset, y_dataset
        """
    return img_txt_to_np("images_dataset", res=res, normalize=normalize,
                         binary=binary, thresh=thresh, verbose=verbose)


def create_RGB2_data(res=(64, 32), normalize=True, thresh=(79//2, 284//2, 0, 255, 0, 107),
                     binary=False, verbose=0):
    """
    creates a dataset from images and a csv file in the "images_from_videos1_dataset" directory.
    This dataset was created by inference of the inferno_labeler.py script on 5 videos.
    :param res: x,y final resolution
    :param normalize: Should the landmarks be normalized
    :param thresh: threshold limits
    :param binary: True for binary pixels, False for grayscale
    :param verbose: 
    :return: x_dataset, y_dataset
    """

    return img_txt_to_np("images_from_videos1_dataset", res=res, normalize=normalize,
                         binary=binary, thresh=thresh, verbose=verbose)


def create_RGB3_data(res=(64, 32), normalize=True, thresh=(79//2, 284//2, 0, 255, 0, 107),
                     binary=False, verbose=0):
    """
    creates a dataset from images and a csv file in the "images_from_videos1_dataset" directory.
    This dataset was created by inference of the inferno_labeler.py script on 5 videos.
    :param res: x,y final resolution
    :param normalize: Should the landmarks be normalized
    :param thresh: threshold limits
    :param binary: True for binary pixels, False for grayscale
    :param verbose:
    :return: x_dataset, y_dataset
    """

    return img_txt_to_np("images_from_videos3_dataset", res=res, normalize=normalize,
                         binary=binary, thresh=thresh, verbose=verbose)


def create_RGB4_data(res=(64, 32), normalize=True, thresh=(79//2, 284//2, 0, 255, 0, 107),
                     binary=False, verbose=0):
    """
    creates a dataset from images and a csv file in the "images_from_videos1_dataset" directory.
    This dataset was created by inference of the inferno_labeler.py script on 5 videos.
    :param res: x,y final resolution
    :param normalize: Should the landmarks be normalized
    :param thresh: threshold limits
    :param binary: True for binary pixels, False for grayscale
    :param verbose:
    :return: x_dataset, y_dataset
    """

    return img_txt_to_np("images_from_camera1_dataset", res=res, normalize=normalize,
                         binary=binary, thresh=thresh, verbose=verbose)


def img_preproccess(image,
                    res=(64, 32),
                    thresh=(79 // 2, 284 // 2, 0, 255, 0, 107),
                    binary=False):
    """
    Takes and image or frame, reduces the resolution, converts to HSV, preforms thresholding,
    and converts to grayscale or binary pixels.
    :param image: image or frame
    :param res: x,y final shape
    :param thresh: threshold limits
    :param binary: blooean, true for binary pixels, false for grayscale
    :return: proccessed image
    """
    low_H, high_H, low_S,  high_S, low_V, high_V = thresh
    resized_frame = cv.resize(image, res, fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    if all(t == 0 for t in thresh):
        frame_new = cv.cvtColor(resized_frame, cv.COLOR_BGR2GRAY)
    else:
        frame_HSV = cv.cvtColor(resized_frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        frame_new = cv.bitwise_and(frame_HSV, frame_HSV, mask=frame_threshold)
        if binary:
            _, frame_new = cv.threshold(frame_new, 0, 255, cv.THRESH_BINARY)
        frame_new = frame_new[:, :, 0]
    frame_array = np.asarray(frame_new)

    return frame_array


def img_txt_to_np(folderpath, res=(64, 32), thresh=(79//2, 284//2, 0, 255, 0, 107),
                  normalize=True, binary=False, verbose=0):
    """
    reads jpg images from a folder and the landmark form a CSV file (each line number coresponds to the image number,
    with the coordiantes as two floats separated by " ". creates from them a dataset of the shapes (N,H,W,1), (N,2)
    :param binary: convert to binary or leave in grayscale
    :param thresh: threshhold to use on HSV format
    :param folderpath:
    :param res: final resolution of images
    :param normalize: Normalize coordinates to 0-1 or not
    :param verbose: 2 - to print out dataset size in the end
    :return: array_list, label_list : lists of image tensors
    """

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(fileDir, folderpath)
    labelpath = os.path.join(path, "labels.txt")
    print("Loading data from:\n", path)
    label_list = []
    array_list = []
    if not os.path.isfile(labelpath):
        print(labelpath)
        raise FileNotFoundError

    with open(labelpath) as f:
        img_count = 0
        for _ in f:
            img_count += 1

    for image_num in tqdm(range(1, img_count+1)):

        image_path = os.path.join(path, "{}.jpg".format(image_num))
        if not os.path.isfile(image_path):
            print(image_path)
            raise FileNotFoundError

        image = cv.imread(image_path, cv.IMREAD_COLOR)

        # img handling
        array_list.append(img_preproccess(image, res=res, thresh=thresh, binary=binary))
        # label handling
        label_line = utils.read_selected_line(labelpath, image_num)
        # original_res = image.shape[:2] # y, x
        original_res = [image.shape[1], image.shape[0]] # x, y
        values = label_line.strip().split(" ")
        values = map(float, values)
        if normalize:
            values = normalize_coord(values, original_res)
        else:  # resize landmarks
            values = normalize_coord(values, [original_res[0]/res[0], original_res[1]/res[1]])
        label_list.append([values[0], values[1]])

    label_list = np.asarray(label_list)
    if verbose >= 2:
        print("label shape:        ", np.shape(label_list), type(label_list))

    array_list = np.asarray(array_list)
    if verbose >= 2:
        print("images array shape: ", np.shape(array_list), type(array_list))

    return array_list, label_list


def view_img_dataset(folderpath="images_dataset"):
    """
    iterates over a folder with numbered jpg images and shows them on screen with the landmark on them
    :param folderpath:
    :return:
    """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(fileDir, folderpath)
    labelpath = os.path.join(path, "labels.txt")
    print("Loading data from folder path:\n", path)
    if not os.path.isfile(labelpath):
        print(labelpath)
        raise FileNotFoundError

    with open(labelpath) as f:
        img_count = 0
        for _ in f:
            img_count += 1

    for image_num in range(1, img_count+1):
        image_path = os.path.join(path, "{}.jpg".format(image_num))
        if not os.path.isfile(image_path):
            print(image_path)
            raise FileNotFoundError
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = np.asarray(image)
        label_line = utils.read_selected_line(labelpath, image_num)
        values = label_line.strip().split(" ")
        values = map(float, values)
        image = cross_annotator(image, values, color=(0, 250, 0), size=2)
        show_img(image)


def flipLR_img_landmark(x_train, y_train):
    """
    flips and adds flipped images to input dataset with one landmark
    :param x_train: images, nparray of shape (N,H,W,1) or (N,H,W,1)
    :param y_train: landmarks, nparray of shape (N,2)
    :return: x_train, y_train, with flipped version appended
    """
    if x_train.ndim == 3:  # for 1 channel images
        x_train = np.expand_dims(x_train, axis=-1)
    if y_train[0, 0] <= 1 and y_train[0, 1] <= 1:
        x_flipedlr = tf.image.flip_left_right(x_train)
        y_flipedlr = np.subtract([[1, 0]], y_train)
        y_flipedlr = np.multiply([[1, -1]], y_flipedlr)
        x_train = np.concatenate((x_train, x_flipedlr), axis=0)
        y_train = np.concatenate((y_train, y_flipedlr), axis=0)

        return x_train, y_train

    else:
        print(y_train[0])
        raise SystemExit("only normalized landmarks are supported.")


def translate_img_landmark_once(x_train, y_train, t_x=3, t_y=2):
    """
    takes a dataset as nparray of shape (N,H,W,1) and landmarks GT of shape (N,2).
    Creates an augmentation of each image as a new dataset which is returned.
    :param x_train: images, nparray of shape (N,H,W,1)
    :param y_train: landmarks, nparray of shape (N,2)
    :param t_x: horizontal translation distrance
    :param t_y: vertical translation distrance
    :return: augmented dataset
    """
    assert x_train.ndim == 4

    img_shape = x_train.shape
    if t_x < 1 and t_y < 1:  # convert to pixels
        shiftX = t_x * img_shape[2]
        shiftY = t_y * img_shape[1]
    else:
        shiftX, shiftY = t_x, t_y
    if y_train[0, 0] < 1 and y_train[0, 1] < 1:  # convert to percent
        trans_vec = np.array([[shiftX/img_shape[2], shiftY/img_shape[1]]])
    else:
        trans_vec = np.array([[shiftX, shiftY]])
    M = np.float32([
        [1, 0, shiftX],
        [0, 1, shiftY]])
    if y_train[0, 0] <= 1 and y_train[0, 1] <= 1:
        shifted_imgs = []
        for image in x_train:
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            shifted_imgs.append(shifted)

        trans_x = np.expand_dims(np.asarray(shifted_imgs), axis = -1)
        trans_y = np.add(y_train, trans_vec)
        return trans_x, trans_y

    else:
        raise SystemExit("only normalized landmarks are supported.")


def translate_img_landmark(x_train, y_train, max_x=8, max_y=2, iterations=4):
    """
    takes a dataset as nparray of shape (N,H,W,1) and landmarks GT of shape (N,2).
    Creates <iterations> augmentations of each image and
    appends them to the input as one dataset which is returned.

    :param x_train: nparray of shape (N,H,W,1)
    :param y_train: nparray of shape (N,2)
    :param max_x: limit of horizontal translation, by pixels
    :param max_y: limit of vertical translation, by pixels
    :param iterations: how many augmentation per image to make
    :return:
    """

    assert x_train.ndim == 4

    if y_train[0, 0] <= 1 and y_train[0, 1] <= 1:
        randx = np.random.randint(-max_x, max_x + 1, size=iterations)
        randy = np.random.randint(-max_y, max_y + 1, size=iterations)
        x_res = []
        y_res = []
        for i in range(iterations):
            trans_x, trans_y = translate_img_landmark_once(x_train, y_train,
                                                           t_x=randx[i], t_y=randy[i])
            # testx, testy = translate_img_landmark_once(trans_x[:5], trans_y[:5])
            # image_utils.plot_example_images(testx, testy,
            #                                 title="trans " + str(i))
            x_res.append(trans_x)
            y_res.append(trans_y)
        for new_datax in x_res:
            x_train = np.concatenate((x_train, new_datax), axis=0)
        for new_datay in y_res:
            y_train = np.concatenate((y_train, new_datay), axis=0)

        return x_train, y_train

    else:
        print(y_train[0])
        cv.imshow("tmp", x_train[0])
        raise SystemExit("only normalized landmarks are supported.")


def view_preproccessed_dataset(data=None,
                               res=(64, 32),
                               fps=5,
                               verbose=2,
                               suffix=None,
                               thresh=(79 // 2, 284 // 2, 0, 255, 0, 107),
                               binary=False):
    """
    loads selected dataset, preproccesses them and creates a video from it
    :param suffix:
    :param verbose:
    :param data:
    :param thresh:
    :param binary:
    :param fps:
    :param res:
    :return:
    """
    if data is None:
        data = ["RGB", "RGB2"]
    data_name = "_".join(data)
    images, labels = prep_data(data=data,
                               binary=binary,
                               verbose=verbose,
                               res=res,
                               thresh=thresh)
    pixel_type = "binary" if binary else "gray"
    name = "{}_{}".format(data_name, pixel_type)
    if suffix is not None:
        if type(str) is not str:
            suffix = str(suffix)
        name += "_" + suffix
    view_dataset(images,
                 labels,
                 name=name,
                 res=res,
                 fps=fps)


def view_dataset(images,
                 labels,
                 name="dataset",
                 res=(64, 32),
                 fps=3,
                 save=True):
    """
    loads selected dataset, preproccesses them and creates a video from it
    :param name:
    :param labels:
    :param images:
    :param fps:
    :param res:
    :return:
    """
    width = res[0]*10
    height = res[1]*10

    cv.namedWindow("output", cv.WINDOW_NORMAL)
    cv.resizeWindow("output", width, height)
    codec = cv.VideoWriter_fourcc(*'XVID') # 'RGBA') # 'MJPG') # 'XVID')
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    output_video_path = os.path.join(fileDir, "preproccessed_video_of_{}.avi".format(name))
    if save:
        out = cv.VideoWriter(output_video_path, codec, fps, (width, height))
        print("saving at:\n{}".format(output_video_path))
    data_size = images.shape[0]
    for img_num in range(data_size):
        image = images[img_num]
        label = labels[img_num]
        image = cross_annotator(image, label, size=2)
        image = cv.resize(image, (width, height), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        if save:
            out.write(image)
        cv.imshow("output", image)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()


def prep_data(data=None,
              binary=True,
              res=(64, 32),
              verbose = 2,
              normalize = True,
              thresh=(79 // 2, 284 // 2, 0, 255, 0, 107)):
    """
    creates and returns nparrays of images and labels, normalized and resized for training
    :param data:
    :param binary:
    :param res:
    :param verbose:
    :param thresh:
    :return:
    """
    known_dataset = ["IR", "RGB", "RGB2", "RGB3", "RGB4"]
    binary_message = "converting to binary pixels.." if binary \
        else "converting to grayscale pixels.."
    print("Loading {} data, and ".format(",".join(data)) + binary_message)

    if data is None:
        data = ["RGB", "RGB2"]
    else:
        data = [string.upper() for string in data]
        for data_name in data:
            if data_name not in known_dataset:
                raise ValueError("Unknown data name: " + data_name)

    img_arrays = []
    landmarks = []

    if "IR" in data:
        images, labels = create_ir_data(thresh=thresh, binary=binary,
                                        res=res, normalize=normalize, verbose=verbose)
        for image in images:
            img_arrays.append(image)
        for label in labels:
            landmarks.append(label)

    if "RGB" in data:
        images, labels = create_RGB_data(thresh=thresh, binary=binary,
                                         res=res, normalize=normalize, verbose=verbose)
        for image in images:
            img_arrays.append(image)
        for label in labels:
            landmarks.append(label)

    if "RGB2" in data:

        images, labels = create_RGB2_data(thresh=thresh, binary=binary,
                                          res=res, normalize=normalize, verbose=verbose)
        for image in images:
            img_arrays.append(image)
        for label in labels:
            landmarks.append(label)

    if "RGB3" in data:

        images, labels = create_RGB3_data(thresh=thresh, binary=binary,
                                          res=res, normalize=normalize, verbose=verbose)
        for image in images:
            img_arrays.append(image)
        for label in labels:
            landmarks.append(label)

    if "RGB4" in data:

        images, labels = create_RGB4_data(thresh=thresh, binary=binary,
                                          res=res, normalize=normalize, verbose=verbose)
        for image in images:
            img_arrays.append(image)
        for label in labels:
            landmarks.append(label)

    img_arrays = np.asarray(img_arrays)
    landmarks = np.asarray(landmarks)
    if verbose > 2:
        image_utils.plot_example_images(np.expand_dims(img_arrays, axis=-1),
                                        landmarks,
                                        title="examples from loaded dataset")
    if img_arrays.ndim == 3:
        img_arrays = np.expand_dims(img_arrays, axis=-1)
    return img_arrays, landmarks


def view_augmentation(data=None,
                      augs=None,
                      binary=True,
                      res=(64, 32),
                      verbose = 2,
                      thresh=(79 // 2, 284 // 2, 0, 255, 0, 107),
                      fps=2):

    if augs is None:
        augs = ["flip", "trans", 2]

    images, labels = prep_data(data=data,
                               binary=binary,
                               res=res,
                               verbose=verbose,
                               thresh=thresh)
    if verbose > 1:
        print("chosen augmentations: " + " ".join(augs))
    if "flip" in augs:
        print("Flipping")
        x_train, y_train = flipLR_img_landmark(images, labels)
        view_dataset(x_train,
                     y_train,
                     name="flipped",
                     res=res,
                     fps=fps)

    if "trans" in augs:
        for aug in augs:
            if aug.isnumeric():

                max_pixel_trans = int(aug)
                print("Translating by {} horizontaly and"
                      " by {} verticaly".format(2 * max_pixel_trans, max_pixel_trans))
                print("WARNING: outofbounds check is not preformed for the landmark")
                if images.ndim == 3:  # for 1 channel images
                    images = np.expand_dims(images, axis=-1)
                x_train, y_train = translate_img_landmark(images, labels,
                                                          max_x=2*max_pixel_trans,
                                                          max_y=max_pixel_trans,
                                                          iterations=1)

                view_dataset(x_train,
                             y_train,
                             name="translated",
                             res=res,
                             fps=fps)


def augmentor(images_, landmarks_, augments_num=4, someof=2, verbose =2):

    new_images, new_labels = [], []

    if landmarks_.ndim != 3:
        landmarks_ = np.expand_dims(landmarks_, axis=1)

    seq = iaa.SomeOf(someof, [
                            iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)),
                            iaa.Fliplr(1),
                            iaa.Crop(percent=(0, 0.1)),
                            # iaa.GaussianBlur(sigma=(0.0, 1.0)),
                            iaa.Affine(rotate=(-5, 5),
                                       scale=(0.9, 1.1),
                                       translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)})
                            ])
    if verbose > 1:
        print(f"preforming {someof} random augmentations on each image, {augments_num} times.")
    for _ in tqdm(range(augments_num)):
        for im in range(images_.shape[0]):
            tmp_landmarks = np.expand_dims(landmarks_[im], axis=1)
            tmp_image = images_[im]
            if verbose > 2:
                print(f"\n#{im} : shape{tmp_image.shape}, landmarks:{tmp_landmarks}")
            image_aug, kpsoi_aug = seq.augment(image=images_[im], keypoints=np.expand_dims(landmarks_[im], axis=1)) # [seq(image=images[0]) for _ in range(8)]
            if verbose > 2:
                print(f"#{im} : shape{image_aug.shape}, landmarks:{kpsoi_aug}")
            new_images.append(image_aug)
            new_labels.append(kpsoi_aug)
        if verbose > 3:
            print("Augmented:")
            ia.imshow(ia.draw_grid(new_images, cols=4, rows=2))

    new_images, new_labels = np.asarray(new_images), np.asarray(new_labels)
    new_labels = np.squeeze(new_labels)
    # new_images = np.asarray(new_images)
    if verbose > 3:
        print(f"new images:\n {new_images.shape} {type(new_images)}")
        print(f"new landmarks:\n {new_labels.shape} {type(new_labels)}")

    return new_images, new_labels


def concat_datasets(images_, labels_, images_b, labels_b):

    images_ = np.concatenate((images_, images_b), axis = 0)
    labels_ = np.concatenate((labels_, labels_b), axis = 0)
    # for image in images_b:
    #     images_.append(image)
    # for label in labels_b:
    #     labels_.append(label)

    return images_, labels_


if __name__ == '__main__':
    resolution_ = config["res"]

    prep_parser = argparse.ArgumentParser(description="perp data")
    prep_parser.add_argument("-d", '--data',
                             nargs='+',
                             default=["RGB"],
                             help='what data to load. available: RGB RGB2 RGB3 IR')
    prep_parser.add_argument('-bin', '--binary',
                             default=False,
                             type=bool,
                             help="'-bin True' for converting data to binary pixels, ignore for False ")
    prep_parser.add_argument("-m", "--mode",
                             default="prep",
                             help="what mode to activate: prep/augs")
    prep_parser.add_argument('-t', '--threshold',
                             nargs='+',
                             default=["0"],
                             help=' threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax) for '
                                  'image preproccessing. or an int for picked values for dictionary')
    prep_parser.add_argument("-a", "--augs",
                             nargs='+',
                             default=["flip", "trans", 2],
                             help="what augmentations to do: "
                                  "flip/"
                                  "trans(if included, you must also add an int to the list for maximal translation")
    prep_args = prep_parser.parse_args()
    # TODO add assertions

    if len(prep_args.threshold) == 6:
        threshold_ = prep_args.threshold
    else:
        assert len(prep_args.threshold) == 1
        threshold_ = config["thresholds"][prep_args.threshold[0]]

    if prep_args.mode == "prep":
        view_preproccessed_dataset(data=prep_args.data,
                                   suffix=None,
                                   res=resolution_,
                                   fps=2,
                                   thresh=threshold_,
                                   binary=prep_args.binary)
    if prep_args.mode == "augs":
        view_augmentation(data=prep_args.data,
                          augs=prep_args.augs,
                          binary=prep_args.binary,
                          res=resolution_,
                          verbose=2,
                          thresh=threshold_,
                          fps=2)
