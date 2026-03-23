import cv2
import numpy as np

# HOG Parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)  # 50% overlap between blocks
cell_size = (8, 8)
nbins = 9

HOG = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def extract_roi(img, cnt, padding=5):
    x, y, w, h = cv2.boundingRect(cnt)
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, img.shape[1])
    y2 = min(y + h + padding, img.shape[0])
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return cv2.resize(roi, (64, 64))  # fixed size for all descriptors


def extract_circle_roi(img, x, y, r, padding=5):
    x1 = max(x - r - padding, 0)
    y1 = max(y - r - padding, 0)
    x2 = min(x + r + padding, img.shape[1])
    y2 = min(y + r + padding, img.shape[0])

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return cv2.resize(roi, (64, 64))


def hu_moments(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 7 values invariant to scale, rotation, translation
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()

    # normalize so the values are not dominated by the first moment (largest)
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return hu


def hog_descriptor(roi):
    # HOG captures gradient structure of the sign
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    features = HOG.compute(gray).flatten()

    return features


def color_histogram(roi, bins=16):
    # histogram on H and S channels only
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist = np.concatenate([h_hist, s_hist])

    # normalizare
    hist = hist / (hist.sum() + 1e-10)

    return hist


def extract_roi_features(roi):
    hu  = hu_moments(roi)    # first 7 vals
    hog = hog_descriptor(roi)
    color = color_histogram(roi)  # last 32 vals

    # concatenate all
    features = np.concatenate([hu, hog, color])

    return features


def extract_features_from_contour(img, cnt):
    roi = extract_roi(img, cnt)
    if roi is None:
        return None

    features = extract_roi_features(roi)
    return features, roi


def extract_features_from_circle(img, x, y, r):
    roi = extract_circle_roi(img, x, y, r)
    if roi is None:
        return None

    features = extract_roi_features(roi)
    return features, roi