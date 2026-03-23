import os
import cv2
import numpy as np
import pickle

from code.color import get_color_masks
from code.preprocessing import preprocess
from code.shape import detect_shapes, encode_shape, encode_color


# -------------------- SVM SETUP --------------------
def build_svm(C=1.0, kernel=cv2.ml.SVM_RBF, gamma=0.5, svm_type=cv2.ml.SVM_C_SVC):
    svm = cv2.ml.SVM_create()
    svm.setType(svm_type)
    svm.setKernel(kernel)
    svm.setC(C)
    svm.setGamma(gamma)
    return svm

# -------------------- EXTRACT FEATURES --------------------
def feature_vector(det):
    base_features = np.array(det["features"], dtype=np.float32)
    shape_features = encode_shape(det["shape"])
    color_features = encode_color(det["color"])

    return np.concatenate([base_features, shape_features, color_features])

# -------------------- TRAINING --------------------
def collect_data(data_dir):
    x, y = [], []

    for folder_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, folder_name)

        if not os.path.isdir(class_dir):
            print(f"Directory not found: {class_dir}")
            continue

        label = int(folder_name)

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = preprocess(img)
            masks = get_color_masks(img)
            _, all_features = detect_shapes(img, masks)

            if not all_features:
                continue

            for det in all_features:
                x.append(feature_vector(det))
                y.append(label)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return x, y


def train_svm(x, y, model_path="trained_svm.xml", C=1.0, gamma=0.5):
    if len(x) == 0:
        raise ValueError("No training data found.")

    svm = build_svm(C=C, gamma=gamma)
    svm.train(x, cv2.ml.ROW_SAMPLE, y)
    svm.save(model_path)

    return svm


# -------------------- GET TRAINED MODEL --------------------
def load_svm(model_path="trained_svm.xml"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    svm = cv2.ml.SVM_load(model_path)

    return svm


def predict_svm(svm, x):
    if x is None or len(x) == 0:
        return []

    _, results = svm.predict(x)

    return results.flatten().astype(int)


def classification_error(y_test, y_pred):
    hits = np.sum(y_pred == y_test)
    misses = np.sum(y_pred != y_test)
    total = len(y_test)

    accuracy = hits / total
    error_rate = misses / total

    mse = np.mean((y_pred - y_test) ** 2)

    print("Hits:", hits)
    print("Misses:", misses)
    print("Accuracy:", accuracy)
    print("Classification error:", error_rate)
    print("MSE:", mse)
