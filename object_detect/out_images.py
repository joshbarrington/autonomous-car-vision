"""
Script to apply object detection to images in test_images directory.
The output images are saved to out_images.
"""

import os               # Python standard library
import cv2              # OpenCV library import

from utils import *

# Name of model to be used.
MODEL_NAME = 'ssd_mobilenet'

# Path to exported frozen detection graph. This is the actual model that is used for
# the object detection.
PATH_TO_CKPT = 'models/' + MODEL_NAME + '/exported_model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')

# Define number of classes.
NUM_CLASSES = 5

# Load label map.
label_map, categories, category_index = load_label_map(PATH_TO_LABELS, NUM_CLASSES)

# Load detection graph, initialise TensorFlow session.
detection_graph, sess = load_frozen_graph(PATH_TO_CKPT)

# Paths of test images.
test_img_paths = [ os.path.join('test_images', 'image{}.jpg'.format(i)) for i in range(1, 5) ]

# Apply Object Detection to each image and save result to new file.
img_num = 1
for image_path in test_img_paths:
    test_img = cv2.imread(image_path, 1)
    objects_detected_image = detect_objects(test_img, sess, detection_graph, category_index)
    cv2.imwrite('out_images/'+MODEL_NAME+'_image'+str(img_num)+'.jpg', objects_detected_image)
    img_num += 1

sess.close()
