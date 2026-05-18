import csv
import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from code.preprocessing import preprocess
from code.shape import detect_shapes, encode_shape, encode_color

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle


# -------------------- EXTRACT FEATURES --------------------
def feature_vector(det):
    base_features = np.array(det["features"], dtype=np.float32)
    shape_features = encode_shape(det["shape"])
    color_features = encode_color(det["color"])

    return np.concatenate([base_features, shape_features, color_features])

# -------------------- TRAINING --------------------
def collect_data(data_dir, train_ratio=0.8):
    x_train, y_train = [], []
    x_test, y_test = [], []

    shape_counts = {}   # ADD THIS
    skipped = 0

    for folder_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, folder_name)

        if not os.path.isdir(class_dir):
            continue

        label = int(folder_name)

        # all images from current class
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".ppm"))
        ]

        # shuffle class images
        random.shuffle(files)

        # split index
        split_idx = int(len(files) * train_ratio)

        train_files = files[:split_idx]
        test_files = files[split_idx:]

        # ---------------- TRAIN ----------------
        for fname in train_files:
            img_path = os.path.join(class_dir, fname)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = preprocess(img)

            _, all_features = detect_shapes(img)

            if not all_features:
                continue

            for det in all_features:
                x_train.append(feature_vector(det))
                y_train.append(label)

                # ADD THIS
                s = det["shape"]
                shape_counts[s] = shape_counts.get(s, 0) + 1

        # ---------------- TEST ----------------
        for fname in test_files:
            img_path = os.path.join(class_dir, fname)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = preprocess(img)

            _, all_features = detect_shapes(img)

            if not all_features:
                continue

            for det in all_features:
                x_test.append(feature_vector(det))
                y_test.append(label)

        # at the end, before return:
    print("Detected shape distribution:", shape_counts)
    print("Skipped images:", skipped)

    return (
        np.array(x_train, dtype=np.float32),
        np.array(y_train, dtype=np.int32),
        np.array(x_test, dtype=np.float32),
        np.array(y_test, dtype=np.int32)
    )


def train_svm(x, y, model_path="trained_svm.pkl", kernel='rbf', C=100, gamma=0.001):
    print('Training SVM...')

    if len(x) == 0:
        raise ValueError("No training data found.")

    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    svm = SVC(C=C, gamma=gamma, kernel=kernel, class_weight='balanced', probability=True)
    svm.fit(x_scaled, y)

    with open(model_path, 'wb') as f:
        pickle.dump({"svm": svm, "scaler": scaler}, f)

    return svm, scaler


def evaluate_params(x, y, model_path="trained_svm.pkl"):
    print('Evaluating SVM parameters...')

    if len(x) == 0:
        raise ValueError("No training data found.")

    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    param_grid = {
        'C':     [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly', 'linear']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=cv, scoring='accuracy', verbose=2)
    grid.fit(x_scaled, y)

    results = grid.cv_results_
    for mean, params in zip(results["mean_test_score"], results["params"]):
        print(f"\nAccuracy: {mean:.4f} -> {params}")

    print(f"\n\nBest params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    best_svm = grid.best_estimator_

    with open(model_path, 'wb') as f:
        pickle.dump({"svm": best_svm, "scaler": scaler}, f)

    return best_svm, scaler

def load_class_names(csv_path):
    """Load class ID → sign name mapping."""
    class_names = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['ClassId'])
            # adjust column name to match your CSV
            name = row.get('Name') or row.get('SignName') or row.get('Shape')
            class_names[class_id] = name
    return class_names

# -------------------- GET TRAINED MODEL --------------------
def load_svm(model_path="trained_svm.pkl"):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data["svm"], data["scaler"]


def predict_svm(svm, X, scaler):
    X = scaler.transform(X)

    probs = svm.predict_proba(X)
    predicted = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    return predicted, confidence

def load_class_shapes(csv_path):
    class_to_shape = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['ClassId'])
            shape = row['Shape']
            class_to_shape[class_id] = shape

    return class_to_shape

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return cm

def plot_confusion_matrix(cm, class_names=None):
    row_sums = cm.sum(axis=1, keepdims=True)

    # Avoid division by zero
    row_sums[row_sums == 0] = 1

    cm = cm.astype('float') / row_sums

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(cm))
    if class_names is not None:
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    else:
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

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


def accuracy_per_shape(y_true, y_pred, class_to_shape):
    shape_stats = {}

    for real, pred in zip(y_true, y_pred):
        shape = class_to_shape[real]

        if shape not in shape_stats:
            shape_stats[shape] = {"correct": 0, "total": 0}

        shape_stats[shape]["total"] += 1

        if real == pred:
            shape_stats[shape]["correct"] += 1

    for shape, stats in shape_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else 0

        print(f"{shape}: {acc:.4f} ({correct}/{total})")
