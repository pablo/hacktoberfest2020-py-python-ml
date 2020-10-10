
# Returns class name for given template id.
from config import SAMPLE_CLASSES


def get_template_name(template_id):
    for sample in SAMPLE_CLASSES:
        if sample.get('class_label') == template_id:
            return sample.get('directory')
    return None

