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
# This code has been modified from https://github.com/naokishibuya/car-finding-lane-line
"""Image processing functions to achieve lane detection"""

import numpy as np  # NumPy import
import cv2          # OpenCV import



def isolate_white_yellow(img):
    """
    Isolate the white and yellow aspects of an image.
    """
    hsl_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    white_lower = np.array([0, 200, 0], dtype="uint8")
    white_upper = np.array([255, 255, 255], dtype="uint8")

    yellow_lower = np.array([20, 100, 100], dtype="uint8")
    yellow_upper = np.array([250, 255, 255], dtype="uint8")

    white_mask = cv2.inRange(hsl_img, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsl_img, yellow_lower, yellow_upper)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    yellow_white_img = cv2.bitwise_and(img, img, mask=mask)
    return yellow_white_img


def binarize(img):
    """
    Apply a binary threshold to an image.
    """
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(grey_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw


def gaussian_blur(img, ksize=5):
    """
    Apply gaussian blur to an image. The default kernel size is 5.
    """
    smooth_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return smooth_img


def region_select(img):
    """
    Select the polygon region of interest of an image.
    """
    rows, cols = np.array(img.shape)
    bottom_left = [cols * 0.1, 0.95 * rows]
    bottom_right = [cols * 0.9, 0.95 * rows]
    top_left = [cols * 0.45, rows * 0.6]
    top_right = [cols * 0.55, rows * 0.6]

    region_points = [np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)]

    mask = np.zeros_like(img)
    ignore_mask_color = 255
    cv2.fillPoly(mask, region_points, ignore_mask_color)

    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def canny_edge(img, min=50, max=100):
    """
    Function to apply the Canny Edge detection algorithm.
    """
    return cv2.Canny(img, min, max)


def hough_lines(img, rho=1, theta=np.pi / 180, threshold=30, min_line_len=10, max_line_gap=8):
    """
    Returns detect lines given an edge detected image.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, min_line_len, max_line_gap)
    return lines


def draw_lines(img, lines):
    """
    Draws lines given line points.
    """
    if lines is None:
        print('No lines to draw')
        return img
    else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        return img


def average_line(lines, image):
    """
    Averages and extrapolates a list of line points to return the points for
    the left lane line and the right lane line.
    """

    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                if slope > 0:
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
                else:
                    left_lines.append((slope, intercept))
                    left_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / \
                np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / \
                 np.sum(right_weights) if len(right_weights) > 0 else None

    y1 = image.shape[0]
    y2 = y1 * 0.65

    left_line_points = line_points(y1, y2, left_lane)
    right_line_points = line_points(y1, y2, right_lane)

    return left_line_points, right_line_points


def line_points(y1, y2, line):
    """
    Given the slop and intercept of a line returns the points of the line.
    """
    if line is not None:
        slope, intercept = line
        if slope > 0 or slope < 0:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            y1 = int(y1)
            y2 = int(y2)
            return [(x1, y1), (x2, y2)]
    else:
        return None


def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=10):
    """
    Draws lines on an image from line points.
    """
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return weighted_img(image, line_image)


def draw_lane_polygon(img, lines):
    """
    Fills a polygon on an image from line points.
    """
    if all(line is not None for line in lines):

        # Polygon points
        p1 = lines[0][0]
        p4 = lines[0][1]
        p2 = lines[1][0]
        p3 = lines[1][1]

        color = [0, 80, 0]
        polygon_points = np.array([p1, p2, p3, p4], np.int32).reshape((-1, 1, 2))

        poly_img = np.zeros_like(img)
        cv2.fillPoly(poly_img, [polygon_points], color)
        return weighted_img(img, poly_img)
    else:
        return img


def weighted_img(img, initial_img, alpha=0.8, beta=1, gamma=0):
    """
    Returns weighted sum of two images.
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)
