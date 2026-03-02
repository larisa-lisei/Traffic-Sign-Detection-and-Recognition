import cv2

def resize_img(img, size=256):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    if scale < 1:
        interp = cv2.INTER_AREA  #downscale
    else:
        interp = cv2.INTER_CUBIC  #upscale

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return resized


def histogram_equalization(img):
    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # Convert back to RGB
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def preprocess(img, resize=256):
    img = resize_img(img, size=resize)
    img = histogram_equalization(img)

    return img
