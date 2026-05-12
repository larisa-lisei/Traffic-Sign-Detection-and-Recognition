import os

import cv2
from sklearn.model_selection import train_test_split

from code.descriptors import hu_moments, extract_roi
from code.model import collect_data, train_svm, predict_svm, load_svm, classification_error, confusion_matrix, \
    load_class_shapes, plot_confusion_matrix, accuracy_per_shape
from code.preprocessing import preprocess
from code.shape import detect_shapes, find_contours


def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DIR = os.path.join(BASE_DIR, '..', 'DATA')
    MODEL = os.path.join(BASE_DIR, 'trained_svm.xml')

    TRAIN_DIR = os.path.join(DIR, 'TRAIN')


    x_train, y_train, x_test, y_test = collect_data(TRAIN_DIR, 0.8)

    # Train
    svm, scaler = train_svm(x_train, y_train, model_path=MODEL, C=1.0, gamma=0.5)

    # Test
    #svm = load_svm(MODEL)
    predicted = predict_svm(svm, x_test, scaler)

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
    img = preprocess(cv2.imread(os.path.join(TRAIN_DIR, '40/40_012.png')))
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