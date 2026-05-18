import cv2
import numpy as np

from code.color import get_color_masks
from code.descriptors import extract_features_from_circle, extract_features_from_contour

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

def encode_shape(shape):
    return np.array(SHAPE_MAP.get(shape, [0, 0, 0, 0]), dtype=np.float32)

def encode_color(color):
    return np.array(COLOR_MAP.get(color, [0, 0, 0, 0]), dtype=np.float32)

# -------------------- CONTOURS --------------------
def find_contours(mask):
    #external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# -------------------- CANNY EDGE DETECTION --------------------
def auto_canny(img, method='otsu', sigma=0.33):
    # blur before canny to suppress noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    if method == "median":
        Th = np.median(blurred)
    elif method == "triangle":
        Th, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_TRIANGLE)
    elif method == "otsu":
        Th, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    else:
        raise Exception("method specified not available!")

    lowTh = (1 - sigma) * Th
    highTh = (1 + sigma) * Th

    edges = cv2.Canny(blurred, lowTh, highTh)
    return edges, highTh

# -------------------- HOUGH TRANSFORM CIRCLES --------------------
def detect_circle(mask):
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,  # precision
        minDist=80,  # min distance between circle centers
        param1=50,  # upper Canny edge threshold
        param2=50,  # votes threshold
        minRadius=10,
        maxRadius=200
    )
    return circles  # each circle is (x, y, radius)

def classify_contour(cnt):
    area = cv2.contourArea(cnt)
    if area < 1000:
        return None

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None

    #create polygon
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    #find vertices
    v = len(approx)

    #polygons
    if v == 3:
        return "triangle"

    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        rect_area = w * h
        extent = area / rect_area

        if extent < 0.65:
            return "triangle"

        return "quadrilateral"

    if v == 8:
        return "octagon"
    return None

def hough_to_contour(circles, shape):
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    largest = max(circles, key=lambda c: c[2])
    x, y, r = largest

    circle_mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), r, 255, -1)

    cnts, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def detect_shapes_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges, _ = auto_canny(gray)
    #cv2.imshow("Canny", edges)

    edge_cnts = find_contours(edges)

    best_cnt = None
    best_shape = None
    for cnt in edge_cnts:
        shape = classify_contour(cnt)
        if shape is not None:
            if best_cnt is None or cv2.contourArea(cnt) > cv2.contourArea(best_cnt):
                best_cnt = cnt
                best_shape = shape

    return best_cnt, best_shape

def detect_shapes_color(mask, img_shape):
    circles = detect_circle(mask)
    circle_cnt = hough_to_contour(circles, img_shape)

    color_cnts = find_contours(mask)
    return integrate_circle_shape(circle_cnt, color_cnts)


def integrate_circle_shape(circle_cnt, shape_cnts):
    best_shape_cnt = None
    for cnt in shape_cnts:
        shape = classify_contour(cnt)
        if shape is not None:
            if best_shape_cnt is None or cv2.contourArea(cnt) > cv2.contourArea(best_shape_cnt):
                best_shape_cnt = cnt

    # prefer polygon
    if best_shape_cnt is not None:
        return best_shape_cnt, classify_contour(best_shape_cnt)

    if circle_cnt is not None:
        return circle_cnt, "circle"

    return None, None


def canny_contour_in_mask(cnt, mask, threshold=0.2):
    # check if enough of the canny contour pixels fall within the color mask
    cnt_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)  # filled contour

    overlap = cv2.bitwise_and(cnt_mask, mask)

    cnt_area = cv2.contourArea(cnt)
    if cnt_area == 0:
        return False

    overlap_area = np.count_nonzero(overlap)
    ratio = overlap_area / cnt_area

    return ratio > threshold


# -------------------- INTEGRATE CANNY + COLOR --------------------
def integrate_edge_color(canny_result, color_result, mask):
    canny_cnt, canny_shape = canny_result
    color_cnt, color_shape = color_result

    # color mask found something
    if color_cnt is not None and color_shape is not None:
        return color_cnt, color_shape

    # only if contour spatially belongs to this color
    if canny_cnt is not None and canny_shape is not None:
        if canny_contour_in_mask(canny_cnt, mask):
            return canny_cnt, canny_shape

    return None, None

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)

    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)

    union = w1*h1 + w2*h2 - inter

    if union == 0:
        return 0

    return inter / union


# -------------------- DETECT SHAPES --------------------
def detect_shapes(img):
    output = img.copy()
    all_features = []

    canny_result = detect_shapes_canny(img)
    masks = get_color_masks(img)

    for color, mask in masks.items():
        #cv2.imshow(color, mask)
        color_result = detect_shapes_color(mask, img.shape)

        final_cnt, final_shape = integrate_edge_color(canny_result, color_result, mask)

        if final_cnt is None or final_shape is None:
            continue

        if final_shape == "circle":
            (x, y), r = cv2.minEnclosingCircle(final_cnt)
            result = extract_features_from_circle(img, int(x), int(y), int(r))
        else:
            result = extract_features_from_contour(img, final_cnt)

        if result is None:
            continue

        features, roi = result
        all_features.append({
            "shape": final_shape,
            "color": color,
            "features": features,
            "roi": roi,
            "bbox": cv2.boundingRect(final_cnt)
        })

        cv2.drawContours(output, [final_cnt], -1, (0, 255, 0), 2)

    return output, all_features



def detect_shapes_canny_video(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges, _ = auto_canny(gray)

    edge_cnts = find_contours(edges)

    results = []

    for cnt in edge_cnts:
        shape = classify_contour(cnt)
        if shape is not None:
            results.append((cnt, shape))

    return results


def detect_shapes_color_video(mask, img_shape):
    circles = detect_circle(mask)
    circle_cnt = hough_to_contour(circles, img_shape)

    color_cnts = find_contours(mask)

    results = []

    # polygons
    for cnt in color_cnts:
        shape = classify_contour(cnt)
        if shape is not None:
            results.append((cnt, shape))

    # circle fallback
    if circle_cnt is not None:
        results.append((circle_cnt, "circle"))

    return results


def detect_shapes_video(img):
    output = img.copy()
    all_features = []

    canny_results = detect_shapes_canny_video(img)
    masks = get_color_masks(img)

    for color, mask in masks.items():
        color_results = detect_shapes_color_video(mask, img.shape)

        combined = []
        for cnt, shape in color_results:
            combined.append((cnt, shape))

        for cnt, shape in canny_results:
            if canny_contour_in_mask(cnt, mask):
                combined.append((cnt, shape))

        for cnt, shape in combined:

            if shape == "circle":
                (x, y), r = cv2.minEnclosingCircle(cnt)
                result = extract_features_from_circle(img, int(x), int(y), int(r))
            else:
                result = extract_features_from_contour(img, cnt)

            if result is None:
                continue

            features, roi = result

            all_features.append({
                "shape": shape,
                "color": color,
                "features": features,
                "roi": roi,
                "bbox": cv2.boundingRect(cnt)
            })

            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

    return output, all_features