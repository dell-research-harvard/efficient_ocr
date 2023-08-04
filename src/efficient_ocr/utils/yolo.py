import json
import os
import cv2

TARGET_LABEL_MAPPING = {0: '0', 1: '1', 2: '0'}
MATCHING_COCO_CATEGORIES = {'line': [2], 'localizer': [0, 1]}

def create_yolo_training_data(coco, target, img_dir = '/data/images', output_dir = r'./data/yolo'):
    if target not in ['line', 'localizer']:
        raise NotImplementedError('Only line and localizer targets are supported for coco-yolo conversion')
    
    with open(coco, 'r') as infile:
        coco = json.load(infile)

    target_id = None
    for cat in coco['categories']:
        if cat['name'] == target:
            target_id = cat['id']
            break

    if target_id is None:
        raise ValueError('Target {} not found in coco file categories'.format(target))
    
    os.makedirs(os.path.join(output_dir, '{}_yolo_data'.format(target)), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '{}_yolo_data/labels'.format(target)), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '{}_yolo_data/images'.format(target)), exist_ok=True)

    for img in coco['images']:
        im = cv2.imread(os.path.join(img_dir, img['file_name']))
        h, w, _ = im.shape
        cv2.imwrite(os.path.join(output_dir, '{}_yolo_data/images'.format(target), img['file_name']), im)

        yolo_labels = []
        for ann in coco['annotations']:
            if ann['image_id'] == img['id'] and ann['category_id'] in MATCHING_COCO_CATEGORIES[target]:
                x, y, width, height = ann['bbox']
                x_center = x + width / 2
                y_center = y + height / 2
                x_center /= w
                y_center /= h
                width /= w
                height /= h
                yolo_labels.append([TARGET_LABEL_MAPPING[ann['category_id']], x_center, y_center, width, height])

        with open(os.path.join(output_dir, '{}_yolo_data/labels'.format(target), img['file_name'].replace('.jpg', '.txt')), 'w') as outfile:
            outfile.write('\n'.join([' '.join([str(x) for x in label]) for label in yolo_labels]))

    return os.path.join(output_dir, '{}_yolo_data'.format(target))
        



def create_yolo_yaml(path, target):
    pass