# MIT License
#
# Copyright (c) 2016 Naoki Shibuya
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# This code has been modified from https://github.com/naokishibuya/car-finding-lane-lines

"""
Lane detector class with detect lanes function. The function applys an image
processing pipeline to detect and highlight lanes on a road image.
"""
from collections import deque   # Python standard library
from lane_detection.detector_tools import *

QUEUE_LENGTH = 10 # Queue for mean lines.


class LaneDetector:

    def __init__(self):
        self.left_lines = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def detect_lanes(self, img):
        # Image processing pipelie
        white_yellow_img = isolate_white_yellow(img)
        binary_img = binarize(white_yellow_img)
        smooth_img = gaussian_blur(binary_img)
        isolated_region = region_select(smooth_img)
        edges = canny_edge(isolated_region)
        lines = hough_lines(edges)
        left_line_point, right_line_point = average_line(lines, img)

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines) > 0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                line = tuple(map(tuple, line))
            return line

        left_line_point = mean_line(left_line_point, self.left_lines)
        right_line_point = mean_line(right_line_point, self.right_lines)

        lane_img = draw_lane_lines(img, (left_line_point, right_line_point))
        lane_img = draw_lane_polygon(lane_img, (left_line_point, right_line_point))

        return lane_img
