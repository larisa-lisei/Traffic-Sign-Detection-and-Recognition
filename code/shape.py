import os
import pickle

import cv2
import numpy as np

from code.preprocessing import preprocess
from code.color import get_color_masks
from code.descriptors import extract_features_from_circle, extract_features_from_contour, hu_moments, extract_roi, \
    hu_moments_contour

SHAPE_MAP = {
    "circle":    [1, 0, 0, 0],
    "triangle":  [0, 1, 0, 0],
    "quadrilateral": [0, 0, 1, 0],
    "octagon":   [0, 0, 0, 1],
}

COLOR_MAP = {
    "red":    [1, 0, 0, 0],
    "blue":   [0, 1, 0, 0],
    "yellow": [0, 0, 1, 0]
}

def create_templates(template_dir, output_dir):
    for fname in os.listdir(template_dir):
        if not fname.endswith((".png", ".jpg")):
            continue

        label = os.path.splitext(fname)[0]  # "circle.png" → "circle"
        path = os.path.join(template_dir, fname)

        img = preprocess(cv2.imread(path))
        masks = get_color_masks(img)
        contours = find_contours(masks["red"])
        cnt = max(contours, key=cv2.contourArea)

        debug_img = cv2.cvtColor(masks["red"], cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 3)

        cv2.imshow(label, debug_img)

        template = hu_moments_contour(cnt)
        save_path = os.path.join(output_dir, f'{label}.pkl')

        with open(save_path, "wb") as f:
            pickle.dump(cnt, f)

def load_templates(dir):
    templates = {}

    for fname in os.listdir(dir):
        if not fname.endswith(".pkl"):
            continue

        label = os.path.splitext(fname)[0]
        path = os.path.join(dir, fname)

        with open(path, "rb") as f:
            templates[label] = pickle.load(f)

    return templates

def match_shape_hu(hu, templates):
    best_label = None
    best_score = float("inf")

    for label, t_hu in templates.items():
        score = cv2.matchShapes(hu, t_hu, cv2.CONTOURS_MATCH_I1, 0)
        print(f"  vs {label}: {score:.4f}")

        if score < best_score:
            best_score = score
            best_label = label

    return best_label, best_score

def encode_shape(shape):
    return np.array(SHAPE_MAP.get(shape, [0, 0, 0, 0]), dtype=np.float32)

def encode_color(color):
    return np.array(COLOR_MAP.get(color, [0, 0, 0, 0]), dtype=np.float32)

# -------------------- CONTOURS --------------------
def find_contours(mask):
    #external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# -------------------- HOUGH TRANSFORM CIRCLES --------------------
def detect_circle(mask, color):
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,  # precision
        minDist=80,  # min distance between circle centers
        param1=50,  # upper Canny edge threshold
        param2=30,  # votes threshold
        minRadius=10,
        maxRadius=200
    )
    return circles  # each circle is (x, y, radius)

def classify_contour(cnt):
    area = cv2.contourArea(cnt)
    if area < 400:
        return None

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None

    #create polygon
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    #find vertices
    v = len(approx)

    #polygons
    if v == 3:
        return "triangle"

    if v == 4:
        return "quadrilateral"

    if v == 8:
        return "octagon"
    return None

# -------------------- DETECT SHAPES --------------------
def detect_shapes(img, masks):
    output = img.copy()
    all_features = []

    for color, mask in masks.items():

        # circle detection using Hough transform
        circles = detect_circle(mask, color)

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)

            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)

                result = extract_features_from_circle(img, x, y ,r)
                if result is None:
                    continue

                features, roi = result

                all_features.append({
                    "shape": 'circle',
                    "color": color,
                    "features": features,
                    "roi": roi,
                })


        # contour based detection for polygons
        contours = find_contours(mask)

        for cnt in contours:
            shape = classify_contour(cnt)
            if shape is None:
                continue

            result = extract_features_from_contour(img, cnt)
            if result is None:
                continue

            features, roi = result

            all_features.append({
                "shape": shape,
                "color": color,
                "features": features,
                "roi": roi,
            })

            # ---------------- DRAW ----------------
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

    return output, all_features


def detect_shapes2(img, masks, templates, threshold=0.5):
    output = img.copy()
    all_features = []

    for color, mask in masks.items():
        contours = find_contours(mask)

        i = 0
        for cnt in contours:
            i = i + 1
            #hu = hu_moments_contour(cnt)
            shape, score = match_shape_hu(cnt, templates)

            debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 3)

            cv2.imshow(f'{color} {i}', debug_img)

            if score > threshold:
                print(f'Score is too big on {color}: {score:.4f}')
                continue

            result = extract_features_from_contour(img, cnt)
            if result is None:
                continue

            features, roi = result

            all_features.append({
                "shape": shape,
                "color": color,
                "features": features,
                "roi": roi,
                "score": score  # Util pentru debugging
            })

            # Desenăm pe imagine
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

    return output, all_features