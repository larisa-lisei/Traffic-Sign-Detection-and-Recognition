import os
import cv2

from code.color import get_color_masks
from code.preprocessing import preprocess
from code.shape import detect_shapes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE_DIR, '..', 'DATA')

img = preprocess(cv2.imread(os.path.join(DIR, '003_000.jpg')))

cv2.imshow('img', img)

masks = get_color_masks(img)

for color, mask in masks.items():
    cv2.imshow(color, mask)

shape, features = detect_shapes(img, masks)
cv2.imshow("Shape detection", shape)

for i, item in enumerate(features):
    print(f'Item {i}:\nshape = {item['shape']}\ncolor = {item['color']}\n\n')

cv2.waitKey(0)
cv2.destroyAllWindows()