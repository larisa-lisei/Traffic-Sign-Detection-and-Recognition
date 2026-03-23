import cv2
import numpy as np

from code.descriptors import extract_features_from_circle, extract_features_from_contour


# -------------------- CONTOURS --------------------
def find_contours(mask):
    #external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# -------------------- ANGLE --------------------
def angle_cos(p0, p1, p2):
    d1 = p0 - p1
    d2 = p2 - p1

    # cos(a) = (d1 * d2) / (norm(d1) * norm(d2))
    return abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10))


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