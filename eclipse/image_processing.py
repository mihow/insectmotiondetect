# image_processing.py
import cv2
import numpy as np


def optical_flow_diff(
    img1, img2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma
):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        0,
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr_flow


def frame_diff(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    binary_diff = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return binary_diff


def image_subtraction(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return diff


def optical_flow_diff_sequence(
    images, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma
):
    flows = []
    for i in range(len(images) - 1):
        gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            0,
        )
        flows.append(flow)

    avg_flow = np.mean(flows, axis=0)
    magnitude, angle = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])
    hsv = np.zeros_like(images[0])
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr_flow
