import os
import re
import tensorflow as tf
from tensorflow import keras
import keras_cv
from training.params import *


def get_labels(labels_dir: str, img_size: int = 416) -> (list, list):
    """gets classes and boxes from labels directory and return tuple of list of both"""
    file_paths = []
    classes = []
    boxes = []
    # Get labels: class ID and x, y, width, height (bounding boxes)
    for root, dirs, files in os.walk(labels_dir):
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
    """gets image paths as list from input directory"""
    paths = []
    for root, dirs, files in os.walk(imgs_dir):
        for file in files:
            img_path = os.path.join(imgs_dir, os.path.splitext(file)[0] + ".jpg")
            paths.append(img_path)
    paths.sort()
    return paths


def get_dataset(dir_labels: str, dir_imgs: str) -> tf.data.Dataset:
    """gets full tf.data.Dataset from labels and image directory"""
    # Use functs from above to get classes, boxes and images paths

    classes, boxes = get_labels(dir_labels)
    img_paths = get_imgs(dir_imgs)

    # Verify
    assert len(classes) == len(boxes) == len(img_paths)

    # Creating data, after converting lists to ragged tensors
    classes_tf = tf.ragged.constant(classes)
    boxes_tf = tf.ragged.constant(boxes)
    paths_tf = tf.ragged.constant(img_paths)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tf, classes_tf, boxes_tf))

    print("✅ Got dataset")

    return dataset


def load_image(image_path):
    """load function to load image from path, to be used in load_dataset function"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    """creates dataset loader, to be mapped on tf.data.Dataset object
    to be used in transform_train_dataset() and transform_val_test_dataset()"""
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


def transform_train_dataset(train_ds: tf.data.Dataset) -> tf.data.Dataset:
    """takes tf.data.Dataset, maps load_dataset function, creates batches, applies augmenter trans"""

    # Data augmentater for the training ds
    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(
                mode="horizontal", bounding_box_format="center_xywh"
            ),
            keras_cv.layers.RandomShear(
                x_factor=0.2, y_factor=0.2, bounding_box_format="center_xywh"
            ),
            keras_cv.layers.JitteredResize(
                target_size=(416, 416),
                scale_factor=(0.75, 1.3),
                bounding_box_format="center_xywh",
            ),
        ]
    )

    # map load_dataset function on train ds, create batches, apply augmenter trans
    train_ds = train_ds.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(BATCH_SIZE * 4)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    print("✅ transformed train dataset")

    return train_ds


def transform_val_test_dataset(val_test_ds: tf.data.Dataset) -> tf.data.Dataset:
    """takes tf.data.Dataset, maps load_dataset function, creates batches, applies augmenter trans"""

    # Data resizer for the validation dataset
    resizing = keras_cv.layers.JitteredResize(
        target_size=(416, 416),
        scale_factor=(0.75, 1.3),
        bounding_box_format="center_xywh",
    )

    # map load_dataset function on train ds, create batches, apply resizing, but NO augmentation
    val_test_ds = val_test_ds.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_test_ds = val_test_ds.shuffle(BATCH_SIZE * 4)
    val_test_ds = val_test_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    val_test_ds = val_test_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    print("✅ transformed test or validation dataset")

    return val_test_ds
