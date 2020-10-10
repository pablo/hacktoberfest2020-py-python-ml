import os
import os.path
from os import path
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from config import WORK_PATH, IMGS_PATH, IMGS_VALID_EXTENSIONS, SIZE, SAMPLE_CLASSES
from util import get_template_name


def get_image_rgb(image_file):
    """extracts features from a given document by applying math operations between pixels."""
    image_features = []
    img = Image.open(image_file)
    img = img.convert('RGB')
    img = img.resize((SIZE, SIZE), Image.BICUBIC)
    for x in range(SIZE):
        rt_h = 0
        gt_h = 0
        bt_h = 0

        rt_v = 0
        gt_v = 0
        bt_v = 0
        for y in range(SIZE):
            r, g, b = img.getpixel((x, y))
            rt_h += r
            gt_h += g
            bt_h += b

            r, g, b = img.getpixel((y, x))
            rt_v += r
            gt_v += g
            bt_v += b
        image_features.append(rt_h)
        image_features.append(gt_h)
        image_features.append(bt_h)
        image_features.append(rt_v)
        image_features.append(gt_v)
        image_features.append(bt_v)
    return image_features


# Extracts features and generates pyc file from which classifiers will be fed.
def generate_features_for_training():
    sample_features = []
    sample_target_classes = []

    i = 1
    for current_sample_class in SAMPLE_CLASSES:
        current_dir = current_sample_class.get('directory')
        current_base_path = IMGS_PATH + '/image_samples/' + current_dir
        current_class_label = current_sample_class.get('class_label')

        print("{}) Reading {}...".format(str(i), current_base_path))

        for f in os.listdir(current_base_path):

            if any([f.lower().endswith(x) for x in IMGS_VALID_EXTENSIONS]):
                current_features = get_image_rgb(current_base_path + '/' + f)
                sample_features.append(current_features)
                sample_target_classes.append(current_class_label)

        with open(WORK_PATH + '/features/simple_classes_featured.txt', 'a') as f2:
            f2.write("{};{}\n".format(current_dir, current_class_label))

        i += 1

    np.save(WORK_PATH + '/features/simple_features', np.array(sample_features))
    np.save(WORK_PATH + '/features/simple_target_classes', np.array(sample_target_classes))


# Returns a model for classification.
def get_classifier():
    # generates features from available training images if needed.
    if not path.exists(WORK_PATH + '/features/simple_features.npy'):
        generate_features_for_training()
    # loads features
    features = np.load(WORK_PATH + '/features/simple_features.npy')
    target_classes = np.load(WORK_PATH + '/features/simple_target_classes.npy')
    # creates new model with generated features
    neigh = KNeighborsClassifier(n_neighbors=1)
    return neigh.fit(features, target_classes)


if __name__ == "__main__":
    # gets classification model
    neigh_model = get_classifier()
    # gets features from a new image.
    sample_features = get_image_rgb(IMGS_PATH + "/prediction_samples/1.png")
    # classifies the new image.
    template_id = int(neigh_model.predict([sample_features])[0])
    print('Documento clasificado como: ' + get_template_name(template_id))