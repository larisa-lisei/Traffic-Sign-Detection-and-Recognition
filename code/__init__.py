import os
import cv2

from code.color import get_color_masks
from code.preprocessing import preprocess
from code.shape import detect_shapes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE_DIR, '..', 'DATA')

img = preprocess(cv2.imread(os.path.join(DIR, '041_0000.png')))

cv2.imshow('img', img)

masks = get_color_masks(img)

for color, mask in masks.items():
    cv2.imshow(color, mask)

shape = detect_shapes(img, masks)
cv2.imshow("Shape detection", shape)

cv2.waitKey(0)
cv2.destroyAllWindows()