# coding: utf-8
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code which has been modified from the notebook available here:
# https://github.com/tensorflow/models/blob/master/research/object_detection

import tensorflow as tf # TensorFlow import
import datetime         # Standard Python Library
import numpy as np      # NumPy library import
import cv2              # OpenCV library import

# Imports from TensorFlow Object Detection API.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_label_map(labels_path, num_classes):
    """
    Load label map, return the categories and index.
    """
    label_map = label_map_util.load_labelmap(labels_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                        max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return label_map, categories, category_index

def load_frozen_graph(checkpoint_path):
    """
    Load a frozen inference graph. Also returns a TensorFlow session.
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
            return detection_graph, sess

def detect_objects(image, sess, graph, category_index):
    """
    Apply object detection. Returns an image with bounding boxes drawn.
    """
    image_np_expanded = np.expand_dims(image, axis=0)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
       [boxes, scores, classes, num_detections],
       feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
       image,
       np.squeeze(boxes),
       np.squeeze(classes).astype(np.int32),
       np.squeeze(scores),
       category_index,
       use_normalized_coordinates=True,
       line_thickness=8)
    return image

class FPS:
    """
    Class to provide utils to calculate frames per second.
    This code has been taken from:
    https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

def fps_label(fps, image_np):
    """
    Function to draw a label indicating the FPS value in the top corner of an image.
    """
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
