import os
import keras_cv
import tensorflow as tf

# Class mapping
class_ids = [
    "10c",
    "10d",
    "10h",
    "10s",
    "2c",
    "2d",
    "2h",
    "2s",
    "3c",
    "3d",
    "3h",
    "3s",
    "4c",
    "4d",
    "4h",
    "4s",
    "5c",
    "5d",
    "5h",
    "5s",
    "6c",
    "6d",
    "6h",
    "6s",
    "7c",
    "7d",
    "7h",
    "7s",
    "8c",
    "8d",
    "8h",
    "8s",
    "9c",
    "9d",
    "9h",
    "9s",
    "Ac",
    "Ad",
    "Ah",
    "As",
    "Jc",
    "Jd",
    "Jh",
    "Js",
    "Kc",
    "Kd",
    "Kh",
    "Ks",
    "Qc",
    "Qd",
    "Qh",
    "Qs",
]

class_mapping = dict(zip(range(len(class_ids)), class_ids))


def create_custom_model():
    """
    Creates custom yolo model. Returns the model.
    """
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xl_backbone_coco")
    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format="center_xywh",
        backbone=backbone,
        fpn_depth=1,
    )
    return yolo


def load_model():
    """
    Loads weightt of a pretrained custom yolo model. Returns the reconstructed model.
    """
    reconstructed_yolo = create_custom_model()
    reconstructed_yolo.load_weights(
        os.path.abspath(os.path.join("models__20230905-0551_model_weights.h5"))
    )
    return reconstructed_yolo


def predict_cards(img, model):
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.resize(image, (416, 416))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)

    y_pred = model.predict(image)

    return y_pred


model = load_model()

image = tf.io.read_file(
    os.path.abspath(os.path.join("inference", "interface", "test.jpg"))
)

y_pred = predict_cards(image, model)


print(y_pred)
