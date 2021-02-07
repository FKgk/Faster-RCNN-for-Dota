import numpy as np
from tqdm import tqdm
import tensorflow as tf

def get_data(input_path, class_mapping):
    classes_count = dict((key, 0) for key in class_mapping.keys())
    all_imgs = {}

    with open(input_path, 'r') as f:
        for line in tqdm(f):
            (filename, x1, y1, x2, y2, class_name) = line.strip().split(',')
            classes_count[class_name] += 1

            if filename not in all_imgs:
                img = tf.io.read_file(filename)
                img = tf.image.decode_image(img)
                (rows, cols) = img.shape[:2]
                
                all_imgs[filename] = {
                    'filepath': filename,
                    'width': cols,
                    'height': rows,
                    'bboxes': []
                }

            all_imgs[filename]['bboxes'].append({
                'class': class_name, 
                'x1': int(round(float(x1))), 'x2': int(round(float(x2))), 
                'y1': int(round(float(y1))), 'y2': int(round(float(y2))), 
            })

    all_data = [value for value in all_imgs.values()]

    return all_data, classes_count
