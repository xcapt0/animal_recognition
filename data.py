# Original dataset https://www.kaggle.com/antoreepjana/animals-detection-images-dataset

import os
import shutil
from tqdm import tqdm
from glob import glob
import re
import cv2


def split_into_folders():
    for folder in ['train', 'test']:
        imgs_path, labels_path = load_data_paths(folder)

        current_path = os.getcwd()
        new_imgs_path = os.path.join(current_path, folder, 'images')
        new_labels_path = os.path.join(current_path, folder, 'labels')

        for path in [new_imgs_path, new_labels_path]:
            if not os.path.isdir(path):
                os.makedirs(path)

        for img, label in tqdm(list(zip(imgs_path, labels_path))):
            if os.path.isfile(label):
                prepare_yolo_label(img, label, new_labels_path)

            if os.path.isfile(img):
                shutil.copy(img, new_imgs_path)


def load_data_paths(data_type):
    imgs = glob(f'animals-detection-images-dataset/{data_type}/*/*.jpg')
    labels = glob(f'animals-detection-images-dataset/{data_type}/*/Label/*.txt')

    assert len(imgs) > 0 and len(imgs) == len(labels), 'Dataset not found'
    return imgs, labels


def prepare_yolo_label(img_path, old_path, new_path):
    labels_dict = load_class_labels()
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(old_path, 'r') as f_old:
        filename = os.path.basename(old_path)
        new_path = os.path.join(new_path, filename)
        old_label = f_old.read()
        old_label_name = re.findall(r'[a-zA-Z\s]+', old_label)[0].strip(' ')
        old_label = re.sub(old_label_name, f'{labels_dict[old_label_name]} ', old_label)
        old_label = old_label.split()

    with open(new_path, 'w') as f_new:
        yolo_label = []

        # for multiple animals
        for i in range(len(old_label[::5])):
            label, xmin, ymin, xmax, ymax = old_label[i * 5:(i * 5) + 5]
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)

            # scaled yolo labels: label x-center y-center width height
            yolo_label.append(label)
            yolo_label.append(str((xmin + xmax) / 2 / w))
            yolo_label.append(str((ymin + ymax) / 2 / h))
            yolo_label.append(str((xmax - xmin) / w))
            yolo_label.append(str((ymax - ymin) / h))

        f_new.write(' '.join(yolo_label))


# {Spider: 0, Lion: 1} etc.
def load_class_labels():
    paths = glob('animals-detection-images-dataset/train/*')
    return {os.path.basename(path).split('.')[0]: i for i, path in enumerate(paths)}
