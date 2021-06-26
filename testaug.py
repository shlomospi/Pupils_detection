import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from prep_data import *
from tqdm import tqdm


# image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")


# ia.imshow(ia.draw_grid(images[:8], cols=4, rows=2))


def augmentor(images_, landmarks_, augments_num=4, someof=2, verbose =2):

    new_images, new_labels = [], []

    if landmarks_.ndim != 3:
        landmarks_ = np.expand_dims(landmarks_, axis=1)

    seq = iaa.SomeOf(someof, [
                            iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)),

                            iaa.Crop(percent=(0, 0.1)),
                            iaa.GaussianBlur(sigma=(0.0, 1.0)),
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
            print(f"\n#{im} : shape{tmp_image.shape}, landmarks:{tmp_landmarks}")
            image_aug, kpsoi_aug = seq.augment(image=images_[im], keypoints=np.expand_dims(landmarks_[im], axis=1)) # [seq(image=images[0]) for _ in range(8)]
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
    images, labels = prep_data(data=["RGB"],
                               binary=False,
                               res=(128, 72),
                               thresh=[0, 0, 0, 0, 0, 0],
                               normalize=False,
                               verbose=0)

    # images = images[:8]
    # labels = labels[:8]
    ne_images, ne_labels = augmentor(images, labels, someof=1, augments_num=2, verbose=2)
    # new_images, new_labels = np.asarray(new_images), np.asarray(new_labels)

    images, labels = concat_datasets(images, labels, ne_images, ne_labels)
    view_dataset(ne_images, ne_labels, save=True, name="imgaug")
    print(f"images:\n {images.shape}")
    print(f"labels:\n {images.shape}")
    ia.imshow(ia.draw_grid(ne_images[:8], cols=4, rows=2))
