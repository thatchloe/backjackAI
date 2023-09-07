import keras_cv
import tensorflow as tf
from tensorflow import keras
import time

from training.params import *
from training.cloud import save_model_to_gcloud


def build_compile_model() -> keras_cv.models.YOLOV8Detector:
    """builds and compiles yolo model returns model"""

    # yolov8 backbone with coco weights
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xl_backbone_coco")

    # Build the model
    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=len(CLASS_MAPPING),
        bounding_box_format="center_xywh",
        backbone=backbone,
        fpn_depth=1,  # Feature Pyramid Network
    )

    # Compile the model with custom adam
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )

    yolo.compile(
        optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
    )

    print("✅ Model built successfully")

    return yolo


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    """Create custom callback, which saves model weights if the
    weights are better than previously according to Coco metrics"""

    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xywh",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images = batch["images"]
            bounding_boxes = batch["bounding_boxes"]

            # Extract "boxes" and "classes" from bounding_boxes
            classes = bounding_boxes["classes"]
            boxes = bounding_boxes["boxes"]

            y_pred = self.model.predict(images, verbose=0)

            # Convert classes and bounding_boxes to a dictionary
            y_true = {"classes": classes, "boxes": boxes}

            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]

        self.best_map = current_map
        self.model.save(self.save_path)
        timestamp = time.strftime("%Y%m%d-%H%M")

        # Create model path
        model_path = os.path.join(
            self.save_path, f"{timestamp}_model_weights_coco_checkpoint.h5"
        )

        # Save model locally and to GCloud
        self.model.save_weights(model_path)
        print("✅ Model checkpoint saved successfully locally")
        save_model_to_gcloud(model_path, BUCKET_NAME)

        return logs
