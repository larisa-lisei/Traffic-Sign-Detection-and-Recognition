import cv2
import numpy as np


def morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # remove noise (erosion + dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def get_color_masks(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # RED
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([155, 25, 40])
    upper_red2 = np.array([180, 255, 255])


    # YELLOW
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # BLUE
    lower_blue = np.array([100, 120, 50])
    upper_blue = np.array([130, 255, 200])

    # MASK
    mask_red = (cv2.inRange(img_hsv, lower_red1, upper_red1) |
                cv2.inRange(img_hsv, lower_red2, upper_red2))
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    mask_red = morphology(mask_red)
    mask_yellow = morphology(mask_yellow)
    mask_blue = morphology(mask_blue)

    return {"red": mask_red, "yellow": mask_yellow, "blue": mask_blue}

