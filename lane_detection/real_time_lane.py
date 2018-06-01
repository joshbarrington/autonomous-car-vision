"""Apply lane detection frame by frame to a video in real time"""

import sys                                  # Python standard library import
sys.path.append('../')
import cv2                                  # OpenCV library import
from lane_detector import LaneDetector
from object_detect.utils import FPS
from object_detect.utils import fps_label


PATH_TO_VIDEO =  '../test_videos/dashcam3.mp4'

video_capture = cv2.VideoCapture(PATH_TO_VIDEO)

fps = FPS().start()
ld = LaneDetector()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if ret:

        height , width , layers =  frame.shape
        half_height = int(height/2)
        half_width = int(width/2)

        frame = cv2.resize(frame, (half_width, half_height))

        detected_lane = ld.detect_lanes(frame)

        fps.update()
        fps.stop()
        fps_label(("FPS: {:.2f}".format(fps.fps())), detected_lane)

        cv2.imshow('Real Time Lane Detection', detected_lane)

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
cv2.destroyAllWindows()
