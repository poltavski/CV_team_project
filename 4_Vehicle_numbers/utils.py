import cv2
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import time
import os

from tensorflow import Tensor, Operation

gpuoptions = tf.GPUOptions(allow_growth=True)
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from PIL import Image
from settings import *
import os.path
from typing import Any, List, Optional, Tuple, Dict, Union

#
# tf.disable_v2_behavior()


class CmpRect:
    def __init__(self, rect):
        self.rect = rect

    def __lt__(self, other):
        # picks up rects
        min_h = min(self.rect[3], other.rect[3]) * COMPARE_COEF
        y_diff = int(abs(self.rect[1] - other.rect[1]) / min_h)
        if y_diff == 0:
            return self.rect[0] < other.rect[0]
        else:
            return self.rect[1] < other.rect[1]


def load_detection_graph(
    model_directory: str = MODELS_PATH, model_name: str = ""
) -> Dict[str, Any]:
    """Loads graph for car, plate and letter detection models.

    Args:
        model_directory: <str> path to .pb model files
        model_name: <str> name of model

    Returns:
        <Dict[str]>
    """
    model_path = os.path.join(model_directory, model_name)

    detection_graph = tf.Graph()

    with detection_graph.as_default():
        # od_graph_def = tf.compat.v1.GraphDef()  # compat above resolves
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
            tf_session = tf.Session(
                config=tf.ConfigProto(gpu_options=gpuoptions), graph=detection_graph
            )
    image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
    detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
    detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
    detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
    num_detections = detection_graph.get_tensor_by_name("num_detections:0")
    result = {
        "tf_session": tf_session,
        "tensor_names": [
            detection_boxes,
            detection_scores,
            detection_classes,
            num_detections,
            image_tensor,
        ],
        "image_tensor": image_tensor,
    }
    return result


def load_classifier_graph(
    model_directory: str = MODELS_PATH, model_name: str = ""
) -> Dict[str, Any]:
    """Loads graph for car brand, color and direction classification models.

    Args:
        model_directory: <str> path to .pb model files
        model_name: <str> name of model

    Returns:
        <Dict[str]>
    """
    model_path = os.path.join(model_directory, model_name)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
            sess = tf.Session(
                config=tf.ConfigProto(gpu_options=gpuoptions), graph=detection_graph
            )
    tensor_output = sess.graph.get_tensor_by_name("dense_2/Softmax:0")
    tensor_input = sess.graph.get_tensor_by_name("input_1:0")
    return [sess, tensor_input, tensor_output]


def best_n_score(arr, n, brand_classes):
    """"""
    top_index = arr[0].argsort()[::-1][:n]
    # print(top_index)
    result = {"car_brand": {}}
    first_time = True
    for i in top_index:
        if first_time:
            top_pred_acc = arr[0][i]
            first_time = False
        result["car_brand"][brand_classes[i]] = round(float(top_pred_acc), 3)
    return result


def rect_intersection(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """Helper function for rectangle intersection

    Args:
        a:
        b:

    Returns:

    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return None
    return x, y, w, h


def rect_area(a: Tuple[int, int, int, int]) -> int:
    """helper function for calculate rectangle area

    Args:
        a: <List[int]> rectangle

    Returns:
        <int> calculated rectangle area
    """
    return a[2] * a[3]
