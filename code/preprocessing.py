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
    # Convert to YCrCb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Apply CLAHE method adapted to contrast on Y channel
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4))
    img_ycrcb[:, :, 0] = clahe.apply(img_ycrcb[:, :, 0])

    # Convert back to RGB
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)


def preprocess(img, resize=256):
    img = resize_img(img, size=resize)
    img = histogram_equalization(img)

    return img
