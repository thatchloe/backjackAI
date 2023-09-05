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
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xl_backbone_coco")
    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format="center_xywh",
        backbone=backbone,
        fpn_depth=1,
    )
    return yolo


def load_model():
    reconstructed_yolo = create_custom_model()
    reconstructed_yolo.load_weights(
        "/Users/sergi/code/seeergiii/blackjack/blackjack/computer_vision/models__20230905-0551_model_weights.h5"
    )
    return reconstructed_yolo


model = load_model()

image = tf.io.read_file(
    "/Users/sergi/code/seeergiii/blackjack/blackjack/computer_vision/computer_vision/interface/test.jpg"
)
image = tf.image.decode_jpeg(image, channels=3)

image = tf.cast(image, tf.float32)
image = tf.image.resize(image, (416, 416))
image = tf.expand_dims(image, axis=0)

y_pred = model.predict(image)

print(y_pred)
