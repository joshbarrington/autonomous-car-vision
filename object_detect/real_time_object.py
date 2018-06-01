"""Apply object detection frame by frame to a video in real time"""

import os               # Python standard library
import cv2              # OpenCV library import
from utils import *


# Name of model to be used.
MODEL_NAME = 'ssd_mobilenet'

# Path to the input video. Change to 0 for webcam.
PATH_TO_VIDEO = '../test_videos/dashcam3.mp4'

# Path to exported frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = 'models/' + MODEL_NAME + '/exported_model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')

# Define number of classes.
NUM_CLASSES = 5

# Load label map.
label_map, categories, category_index = load_label_map(PATH_TO_LABELS, NUM_CLASSES)

# Load detection graph, initialise TensorFlow session.
detection_graph, sess = load_frozen_graph(PATH_TO_CKPT)

# Load video source
video_capture = cv2.VideoCapture(PATH_TO_VIDEO)

# Initialise FPS
fps = FPS().start()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if ret:
        height , width , layers =  frame.shape
        half_height = int(height/2)
        half_width = int(width/2)

        frame = cv2.resize(frame, (half_width, half_height))

        # Apply Object Detection
        detected_image = detect_objects(frame, sess, detection_graph, category_index)
        # Update FPS counter
        fps.update()
        # Stop FPS
        fps.stop()

        fps_label(("FPS: {:.2f}".format(fps.fps())), detected_image)
        # Show image witb object detection
        cv2.imshow('Real Time Object Detection', detected_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
            print('[INFO] number of frames: {:d}'.format(fps._numFrames))
            print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
            break
    else:
        video_capture.release()
        if fps._end is None:
            print("[INFO] No frame loaded.")
        else:
            print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
            print('[INFO] number of frames: {:d}'.format(fps._numFrames))
            print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

sess.close()
cv2.destroyAllWindows()
