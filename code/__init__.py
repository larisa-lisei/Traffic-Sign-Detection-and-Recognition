import os

import cv2

from code.color import get_color_masks
from code.descriptors import hu_moments, extract_roi
from code.model import collect_data, train_svm, predict_svm, load_svm, classification_error
from code.preprocessing import preprocess
from code.shape import detect_shapes, find_contours, create_templates, load_templates, detect_shapes2


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DIR = os.path.join(BASE_DIR, '..', 'DATA')
    MODEL = os.path.join(BASE_DIR, 'trained_svm.xml')

    TRAIN_DIR = os.path.join(DIR, 'TRAIN')
    HU_TRAIN_DIR = os.path.join(DIR, 'Hu_moments_training')
    HU_OUTPUT_DIR = os.path.join(DIR, 'ShapeTemplates')

    # create_templates(HU_TRAIN_DIR, HU_OUTPUT_DIR)

    shape_templates = load_templates(HU_OUTPUT_DIR)

    img = preprocess(cv2.imread(os.path.join(TRAIN_DIR, '0/0_001.png')))
    cv2.imshow('img', img)
    masks = get_color_masks(img)

    for color, mask in masks.items():
        cv2.imshow(color, mask)

    shape, features = detect_shapes2(img, masks, shape_templates)
    cv2.imshow("Shape detection", shape)

    for i, item in enumerate(features):
        print(f'Item {i}:\nshape = {item['shape']}\ncolor = {item['color']}\n\n')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    if not os.path.exists(MODEL):
        x, y = collect_data(TRAIN_DIR)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, shuffle=True
        )
        svm = train_svm(x_train, y_train, model_path=MODEL, C=1.0, gamma=0.5)
    
        predicted = predict_svm(svm, x_test)
        classification_error(y_test, predicted)
    else:
        svm = load_svm(MODEL)
    
        x, y = collect_data(TRAIN_DIR)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, shuffle=True
        )
        predicted = predict_svm(svm, x_test)
    
        for real, pred in zip(y_test, predicted):
            if real != pred:
                print(f"real={real}, pred={pred}")
    
        classification_error(y_test, predicted)
    '''


    '''
    img = preprocess(cv2.imread(os.path.join(TRAIN_DIR, '22/020_1_0001.png')))
    
    cv2.imshow('img', img)
    
    masks = get_color_masks(img)
    
    for color, mask in masks.items():
        cv2.imshow(color, mask)
    
    shape, features = detect_shapes(img, masks)
    cv2.imshow("Shape detection", shape)
    
    for i, item in enumerate(features):
        print(f'Item {i}:\nshape = {item['shape']}\ncolor = {item['color']}\n\n')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

if __name__ == '__main__':
    main()