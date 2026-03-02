import cv2
import numpy as np
import math

def find_contours(mask):
    #external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

#calculate angle
def angle_cos(p0, p1, p2):
    d1 = p0 - p1
    d2 = p2 - p1

    # cos(a) = (d1 * d2) / (norm(d1) * norm(d2))
    return abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10))

def classify_contour(cnt):
    area = cv2.contourArea(cnt)
    if area < 400:
        return None

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None

    #create polygon
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

    #find vertices
    v = len(approx)

    # circularity = (4PI * A) / P^2
    # perfect circle => circularity = 1
    circularity = 4 * math.pi * area / (peri * peri)

    if v > 8 and circularity > 0.70:
        return "circle"

    #polygons
    if v == 3:
        return "triangle"

    if v == 4:
        #get (x, y) for polygon
        pts = approx.reshape(4, 2).astype(np.float32)

        # compute angles for each corner
        cosines = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            p2 = pts[(i + 2) % 4]
            cosines.append(angle_cos(p0, p1, p2))

        #cos(90) = 0 => min(cos) ~= 90 degrees
        #max(cos) = worst case scenario
        max_cos = max(cosines)

        # get width and height of shape
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        (_, _), (w, h), angle = cv2.minAreaRect(cnt)
        angle = abs(angle)

        if max_cos < 0.3 and 0.85 <= aspect_ratio <= 1.15:
            # diamond = rotated rectangle with ~45 degrees
            if 30 <= angle <= 60:
                return "diamond"
            return "square"

        if 30 <= angle <= 60:
            return "diamond"
        return "rectangle"

    if v == 8:
        return "octagon"

    print("Unknown shape")
    return None

def detect_shapes(img, masks):
    output = img.copy()

    for color, mask in masks.items():
        contours = find_contours(mask)

        for cnt in contours:
            shape = classify_contour(cnt)
            if shape is None:
                continue

            print(f"{color} {shape}")

            #draw green contour
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

    return output