import os

from config import IMGS_PATH, SAMPLE_CLASSES, IMGS_VALID_EXTENSIONS
from trainer import get_classifier, get_image_rgb
from util import get_template_name

if __name__ == "__main__":
    # gets classification model
    neigh_model = get_classifier()

    total = 0
    correct = 0

    for current_sample_class in SAMPLE_CLASSES:
        current_dir = current_sample_class.get('directory')
        current_base_path = IMGS_PATH + '/prediction_samples/' + current_dir
        current_class_label = current_sample_class.get('class_label')

        for f in os.listdir(current_base_path):
            if any([f.lower().endswith(x) for x in IMGS_VALID_EXTENSIONS]):
                total += 1
                current_features = get_image_rgb(current_base_path + '/' + f)
                calculated_template_id = int(neigh_model.predict([current_features])[0])
                print(f'Documento clasificado como: {get_template_name(calculated_template_id)} - Se esperaba: {get_template_name(current_sample_class.get("class_label"))}')
                if calculated_template_id == current_sample_class['class_label']:
                    correct += 1
                else:
                    print(f"ERROR! {calculated_template_id} != {current_sample_class['class_label']}")


    print(f"Efectividad del clasificador: {(float(correct)/float(total))*100}%")