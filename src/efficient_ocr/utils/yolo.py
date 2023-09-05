import json
import os
import cv2


def create_yolo_training_data(
        coco, 
        img_dir, 
        target, 
        output_dir,
        char_only,
        split = [0.8, 0.2]):
    
    if target not in ['line', 'localizer']:
        raise NotImplementedError('Only line and localizer targets are supported for coco-yolo conversion')
    
    if type(coco) == str:
        with open(coco, 'r') as infile:
            coco = json.load(infile)
    elif isinstance(coco, dict):
        pass
    else:
        raise ValueError('coco must be a path to a coco json file or a coco dict')
    
    cat_dict = {cat['name']: cat['id'] for cat in coco['categories']}

    if target == 'localizer':
        if char_only:
            coco_cats = [v for k, v in cat_dict.items() if 'char' in k]
        else:
            coco_cats = [v for k, v in cat_dict.items() if 'char' in k or 'word' in k]
    elif target == 'line':
        coco_cats = [v for k, v in cat_dict.items() if 'line' in k]
    else:
        raise NotImplementedError
    
    target_label_mapping = {int(cat): str(idx) for idx, cat in enumerate(coco_cats)}
    
    os.makedirs(os.path.join(output_dir, '{}_yolo_data'.format(target)), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '{}_yolo_data/train'.format(target)), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '{}_yolo_data/val'.format(target)), exist_ok=True)
    for subdir in ['images', 'labels']:
        for partition in ['train', 'val']:
            os.makedirs(os.path.join(output_dir, '{}_yolo_data'.format(target), partition, subdir), exist_ok=True)

    for i, img in enumerate(coco['images']):
        if i < split[0] * len(coco['images']):
            partition_dir = os.path.join(output_dir, '{}_yolo_data/train'.format(target))
        else:
            partition_dir = os.path.join(output_dir, '{}_yolo_data/val'.format(target))

        im = cv2.imread(os.path.join(img_dir, img['file_name']))
        h, w, _ = im.shape

        cv2.imwrite(os.path.join(partition_dir, 'images', img['file_name']), im)

        yolo_labels = []
        for ann in coco['annotations']:
            if ann['image_id'] == img['id'] and ann['category_id'] in coco_cats:
                x, y, width, height = ann['bbox']
                x_center = x + width / 2
                y_center = y + height / 2
                x_center /= w
                y_center /= h
                width /= w
                height /= h
                yolo_labels.append([target_label_mapping[ann['category_id']], x_center, y_center, width, height])

        with open(os.path.join(partition_dir, 'labels', img['file_name'].replace('.png', '.txt')), 'w') as outfile:
            outfile.write('\n'.join([' '.join([str(x) for x in label]) for label in yolo_labels]))

    path = os.path.join(output_dir, '{}_yolo_data'.format(target))

    names = ['line'] if target == 'line' else ['char', 'word']

    with open(os.path.join(path, 'data.yaml'), 'w') as outfile:
        outfile.write('train: {}\n'.format(os.path.abspath(os.path.join(path, 'train', 'images'))))
        outfile.write('val: {}\n'.format(os.path.abspath(os.path.join(path, 'val', 'images'))))
        outfile.write('nc: {}\n'.format(len(target_label_mapping)))
        outfile.write('names: {}\n'.format(names))

    return os.path.join(path, 'data.yaml')
        

def create_yolo_yaml():
    pass