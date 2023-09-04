import os
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.config import set_visible_devices, get_visible_devices
from tqdm.auto import tqdm
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

from interface.params import *

SPLIT_RATIO = 0.2
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCH = 128
GLOBAL_CLIPNORM = 10.0
IMAGE_SIZE = 416
USE_GPU = True


def get_labels(labels_dir: str, img_size: int = 416) -> (list, list):
    """Get classes and bounding boxes each as a list from directory containing .txt files"""
    file_paths = []
    classes = []
    boxes = []
    # Get labels: class ID and x, y, width, height (bounding boxes)
    for _, _, files in os.walk(labels_dir):
        for file in files:
            # Get path for each label .txt file
            label_path = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")
            file_paths.append(label_path)

        # Sort paths by picture name
        pattern = r"(\d+)_jpg\.rf"
        file_paths.sort(key=lambda x: int(re.findall(pattern, x)[0]))

        for file in file_paths:
            # Open label .txt file
            with open(file, "r") as label_file:
                label_lines = label_file.readlines()

                img_classes = []
                img_boxes = []

                # Parse label lines
                for line in label_lines:
                    parts = line.split()
                    single_class = int(parts[0])  # First value is the class label
                    img_classes.append(single_class)

                    single_box = [
                        float(x) * img_size for x in parts[1:]
                    ]  # Rest are the boxes (x,y and height,width)
                    img_boxes.append(single_box)

                classes.append(img_classes)
                boxes.append(img_boxes)
    return classes, boxes


def get_imgs(imgs_dir: str) -> list:
    """Get all image paths as list from directory containing images"""
    paths = []
    for _, _, files in os.walk(imgs_dir):
        for file in files:
            img_path = os.path.join(imgs_dir, os.path.splitext(file)[0] + ".jpg")
            paths.append(img_path)
    paths.sort()
    return paths


def get_test_data(labels_dir: str, imgs_dir: str) -> tf.data.Dataset:
    """Get complete test data as tensorflow Dataset object, calls get_labels and get_imgs functions"""
    classes, boxes = get_labels(labels_dir)
    img_paths = get_imgs(imgs_dir)

    classes_tf = tf.ragged.constant(classes)
    boxes_tf = tf.ragged.constant(boxes)
    paths_tf = tf.ragged.constant(img_paths)

    data = tf.data.Dataset.from_tensor_slices((paths_tf, classes_tf, boxes_tf))

    return data


def load_image(image_path):
    """load image from image path"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image  # Loading data functions


def load_dataset(image_path, classes, bbox):
    """load dataset from"""
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": image, "bounding_boxes": bounding_boxes}
