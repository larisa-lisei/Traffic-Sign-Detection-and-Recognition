import os

import cv2
import numpy as np

from code.descriptors import hu_moments, extract_roi
from code.model import collect_data, train_svm, predict_svm, load_svm, classification_error, confusion_matrix, \
    load_class_shapes, plot_confusion_matrix, accuracy_per_shape, load_class_names, feature_vector
from code.preprocessing import preprocess, histogram_equalization
from code.shape import detect_shapes, find_contours, detect_shapes_video


def load_label_templates(DATA_DIR, classes):

    templates = {}
    for class_id, label_name in classes.items():
        folder = os.path.join(DATA_DIR, str(class_id))

        img_found = None
        if os.path.isdir(folder):
            images = sorted(os.listdir(folder))

            if len(images) == 0:
                continue

            img_path = os.path.join(folder, images[0])
            img_found = cv2.imread(img_path)

        if img_found is not None:
            templates[class_id] = img_found

    return templates


def demo_video(video_path, svm, scaler, class_templates, output_path=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = histogram_equalization(frame)
        all_features = detect_shapes_video(img)

        if all_features:
            x = np.array([feature_vector(det) for det in all_features], dtype=np.float32)
            predicted, confidence = predict_svm(svm, x, scaler)
            print(confidence)

            for i, (det, pred_label) in enumerate(zip(all_features, predicted)):
                if confidence[i] < 0.6:
                    continue

                template = class_templates.get(int(pred_label))

                if "bbox" in det:
                    bx, by, bw, bh = det["bbox"]

                    # draw bounding box
                    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

                    # draw template next to the box
                    if template is not None:
                        sign_img = cv2.resize(template, (100, 100))

                        x2 = bx + bw + 10
                        y2 = by

                        h_frame, w_frame = img.shape[:2]

                        # avoid going outside frame
                        if x2 + 100 > w_frame:
                            x2 = bx - 110
                        if y2 + 100 > h_frame:
                            y2 = h_frame - 100

                        img[y2:y2 + 100, x2:x2 + 100] = sign_img

        # resize back to original dimensions before writing
        out_frame = cv2.resize(img, (w, h))

        cv2.imshow("Demo", out_frame)


        if writer:
            writer.write(out_frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DIR = os.path.join(BASE_DIR, '..', 'DATA')
    MODEL = os.path.join(BASE_DIR, 'trained_svm.xml')
    TRAIN_DIR = os.path.join(DIR, 'TRAIN')
    LABELS = os.path.join(DIR, 'labels.csv')

    VIDEO = os.path.join(DIR, 'demo3.mp4')

    #svm, scaler = load_svm(MODEL)
    classes = load_class_names(LABELS)
    class_templates = load_label_templates(TRAIN_DIR, classes)

    #demo_video(VIDEO, svm, scaler, class_templates, output_path=os.path.join(DIR, 'output.mp4'))


    x_train, y_train, x_test, y_test = collect_data(TRAIN_DIR, 0.8)

    # Train
    #svm, scaler = train_svm(x_train, y_train, model_path=MODEL)

    # Test

    svm, scaler = load_svm(MODEL)
    predicted, confidence = predict_svm(svm, x_test, scaler)

    # Evaluation metrics
    classes = load_class_shapes(os.path.join(DIR, 'labels.csv'))
    cm = confusion_matrix(y_test, predicted, len(classes))
    plot_confusion_matrix(cm, classes)

    accuracy_per_shape(y_test, predicted, classes)


    for real, pred in zip(y_test, predicted):
        if real != pred:
            print(f"real={real}, pred={pred}")

    classification_error(y_test, predicted)


    '''
    img = preprocess(cv2.imread(os.path.join(TRAIN_DIR, '41/41_005.png')))
    cv2.imshow('img', img)

    shape, features = detect_shapes(img)

    cv2.imshow("Shape detection", shape)

    for i, item in enumerate(features):
        print(f'Item {i}:\nshape = {item['shape']}\ncolor = {item['color']}\n\n')


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

if __name__ == '__main__':
    main()