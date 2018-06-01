"""
Script to apply the object detection and lane detection modules to each frame of
a video, and write the results to a new video.

"""

from lane_detection.detector_tools import *
from lane_detection.lane_detector import LaneDetector
from object_detect.utils import *

from moviepy.editor import VideoFileClip

# Name of the model to be used.
MODEL_NAME = 'faster_rcnn_resnet50'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'object_detect/models/' + MODEL_NAME + '/exported_model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object_detect/data/label_map.pbtxt'

NUM_CLASSES = 5

label_map, categories, category_index = load_label_map(PATH_TO_LABELS, NUM_CLASSES)

VIDEO_NAME = 'dashcam1.mp4'
OUTPUT_NAME = 'lane_object_frcnn.mp4'

PATH_TO_VIDEO = 'test_videos/' + VIDEO_NAME
PATH_TO_OUTPUT = 'output_videos/' + OUTPUT_NAME

# The way the object detection is applied has been reorded to provide a Function
# which requires a single image parameter. This is to enable the fl_image function
# to be used when processing the video frames.
def detect_objects(image):
    image_np_expanded = np.expand_dims(image, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
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

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
        lane_detector = LaneDetector()
        clip = VideoFileClip(PATH_TO_VIDEO)
        processed = clip.fl_image(detect_objects)
        processed = processed.fl_image(lane_detector.detect_lanes)
        processed.write_videofile(PATH_TO_OUTPUT, audio=False)
