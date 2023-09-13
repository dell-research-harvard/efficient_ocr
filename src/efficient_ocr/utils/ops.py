import torch
import torch.nn as nn
import time
import torchvision
import copy
import numpy as np
import cv2
import os
import json

DEFAULT_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DEFAULT_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def create_batches(data, batch_size = 64, transform = None):
    """Create batches for inference"""

    batches = []
    batch = []
    for i, d in enumerate(data):
        if d is not None:
            batch.append(d)
        else:
            batch.append(np.zeros((33, 33, 3), dtype=np.int8))
        if (i+1) % batch_size == 0:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)
    return [b for b in batches]

def get_onnx_input_name(model):
    input_all = [node.name for node in model.graph.input]
    input_initializer =  [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))
    return net_feed_input[0]

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

def yolov8_non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = False  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

    return output

def yolov5_non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  ):

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

    return output

COCO_TEMPLATE = {'images': [],
                 'annotations': [],
                 'categories': [{'id': 0, 'name': 'char'}, {'id': 1, 'name': 'word'}, {'id': 2, 'name': 'line'}]}

ANNOTATION_TEMPLATE = {'segmentation': [], 'area': None, 'bbox': [], 'iscrowd': 0, 'image_id': None, 'category_id': None, 'id': None}

def make_coco_from_effocr_result(result, imgs, save_path = None, skip_lines = False, char_only = False):
    '''
    Takes in an effocr result in the format:
        { bbox_idx: {
                        line_idx: {
                            'words': [(word_img, (y0, x0, y1, x1)), ...],
                            'chars': [(char_img, (y0, x0, y1, x1)), ...],
                            'overlaps': [[char_idx, char_idx, ...], ...],
                            'para_end': bool,
                            'bbox': (y0, x0, y1, x1),
                            'final_puncs': [word_end, ...],
                            'word_preds': [word_pred, ...]
                        },
                        ...
                    },
                    ...
        }}

    And produces a coco format annotation
    '''
    coco = copy.deepcopy(COCO_TEMPLATE)

    for i, img in enumerate(imgs):
        cv2.imwrite(f'./data/images/{i}.png', img)
        coco['images'].append({'id': i, 'file_name': f'{i}.png', 'height': img.shape[0], 'width': img.shape[1], 'text': result[i].text})

    for img in coco['images']:
        for line_idx in result[img['id']].preds.keys():
            line_anno = copy.deepcopy(ANNOTATION_TEMPLATE)
            line_anno['image_id'] = img['id']
            line_anno['category_id'] = 2
            line_anno['id'] = len(coco['annotations'])

            line_y0, line_x0, line_y1, line_x1 = result[img['id']].preds[line_idx]['bbox']
            line_anno['bbox'] = [line_x0, line_y0, line_x1 - line_x0, line_y1 - line_y0]
            line_anno['segmentation'] = [[line_x0, line_y0, line_x1, line_y0, line_x1, line_y1, line_x0, line_y1]]
            line_text = ' '.join(result[img['id']].preds[line_idx]['word_preds'])
            line_anno['text'] = line_text
            line_anno['area'] = (line_x1 - line_x0) * (line_y1 - line_y0)
            line_text = line_text.replace(' ', '')
            coco['annotations'].append(line_anno)

            for i, word in enumerate(result[img['id']].preds[line_idx]['words']):
                word_anno = copy.deepcopy(ANNOTATION_TEMPLATE)
                word_anno['image_id'] = img['id']
                word_anno['category_id'] = 1
                word_anno['id'] = len(coco['annotations'])

                y0, x0, y1, x1 = word[1]
                y0 += line_y0; x0 += line_x0; y1 += line_y0; x1 += line_x0
                word_anno['bbox'] = [x0, y0, x1 - x0, y1 - y0]
                word_anno['segmentation'] = [[x0, y0, x1, y0, x1, y1, x0, y1]]
                word_anno['text'] = result[img['id']].preds[line_idx]['word_preds'][i]
                word_anno['area'] = (x1 - x0) * (y1 - y0)
                coco['annotations'].append(word_anno)

            for i, char in enumerate(result[img['id']].preds[line_idx]['chars']):
                char_anno = copy.deepcopy(ANNOTATION_TEMPLATE)
                char_anno['image_id'] = img['id']
                char_anno['category_id'] = 0
                char_anno['id'] = len(coco['annotations'])

                y0, x0, y1, x1 = char[1]
                y0 += line_y0; x0 += line_x0; y1 += line_y0; x1 += line_x0
                char_anno['bbox'] = [x0, y0, x1 - x0, y1 - y0]
                char_anno['segmentation'] = [[x0, y0, x1, y0, x1, y1, x0, y1]]
                char_anno['area'] = (x1 - x0) * (y1 - y0)
                try:
                    char_anno['text'] = line_text[i]
                except IndexError:
                    char_anno['text'] = ''

                coco['annotations'].append(char_anno)

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(coco, f, indent=4)


def visualize_effocr_result(imgs, annotations_path, save_path, to_display = 'full'):
    if to_display is None:
        to_display = 'full'
        
    with open(annotations_path, 'r') as infile:
        coco = json.load(infile)

    for img_idx, img in enumerate(imgs):
        img_coco = coco['images'][img_idx]

        # Create a blank white canvas with height and width 3x the original image's
        canvas = np.zeros((int(img_coco['height'] * 2.5), int(img_coco['width'] * 3), 3), dtype=np.uint8)
        canvas.fill(255)

        # Paste the image into the canvas at .5x image height and .5x image width
        canvas[img_coco['height'] // 4 :img_coco['height'] // 4 + img_coco['height'], img_coco['width'] // 3:img_coco['width'] // 3 + img_coco['width']] = img

        cat_mapping = {cat['name']: cat['id'] for cat in coco['categories']}

        # Create a blank canvas for the line texts, the same size as the image
        text_canvas = np.zeros((img_coco['height'], img_coco['width'], 3), dtype=np.uint8)
        text_canvas.fill(255)

        # Assemble a list of all the line annotations associated with this image
        line_annos = [anno for anno in coco['annotations'] if anno['image_id'] == img_coco['id'] and anno['category_id'] == cat_mapping['line']]

        # Sort the line annotations by y coordinate
        line_annos.sort(key=lambda x: x['bbox'][1])

        # For each line annotation, paste the text into the text at the y location of the line on the text_canvas
        for line_anno in line_annos:
            line_y0, line_x0, line_y1, line_x1 = line_anno['bbox']
            line_text = line_anno['text']
            cv2.putText(text_canvas, line_text, (0, line_x0 + line_x1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)

        # Paste the text_canvas into the main canvas, with the top left corner at (.5x image height, 2x image width))
        # This is so that the text is to the right of the image, with half an image width between text and image
        canvas[img_coco['height'] // 4:img_coco['height'] // 4 + img_coco['height'], int(img_coco['width'] * (5/3)):int(img_coco['width'] * (5/3)) + img_coco['width']] = text_canvas
        
        # Create three copies of the image, each one with bounding boxes annotations from lines, words, and characters
        line_canvas = copy.deepcopy(img)
        word_canvas = copy.deepcopy(img)
        char_canvas = copy.deepcopy(img)

        for anno in coco['annotations']:
            if anno['image_id'] == img_coco['id']:
                x0, y0, x1, y1 = anno['bbox']
                if anno['category_id'] == cat_mapping['line']:
                    cv2.rectangle(line_canvas, (x0, y0), (x0 + x1, y0 + y1), (0, 0, 255), 2)
                elif anno['category_id'] == cat_mapping['word']:
                    cv2.rectangle(word_canvas, (x0, y0), (x0 + x1, y0 + y1), (0, 255, ), 1)
                elif anno['category_id'] == cat_mapping['char']:
                    cv2.rectangle(char_canvas, (x0, y0), (x0 + x1, y0 + y1), (255, 0, 0), 1)

        # Paste the three canvases into the main canvas, with the top height at (1.5 image height) and top corners evenly spaced, starting at .5 image width
        # Scale down all three images to 2/3 of their original size
        line_canvas_ = cv2.resize(line_canvas, (2 * img_coco['width'] // 3, 2 * img_coco['height'] // 3))
        word_canvas_ = cv2.resize(word_canvas, (2 * img_coco['width'] // 3, 2 * img_coco['height'] // 3))
        char_canvas_ = cv2.resize(char_canvas, (2 * img_coco['width'] // 3, 2 * img_coco['height'] // 3))
        y_top = int(1.5 * img_coco['height'])
        x_left = int(.25 * img_coco['width'])
        canvas[y_top:y_top + line_canvas_.shape[0], x_left : x_left + line_canvas_.shape[1]] = line_canvas_
        x_left = int(1.166 * img_coco['width'])
        canvas[y_top:y_top + word_canvas_.shape[0], x_left : x_left + word_canvas_.shape[1]] = word_canvas_
        x_left = int(2.083 * img_coco['width'])
        canvas[y_top:y_top + char_canvas_.shape[0], x_left : x_left + char_canvas_.shape[1]] = char_canvas_

        if to_display == 'full':
            to_save = canvas
        elif to_display == 'line':
            to_save = line_canvas
        elif to_display == 'word':
            to_save = word_canvas
        elif to_display == 'char':
            to_save = char_canvas
        else:
            raise ValueError('to_display must be one of "full", "line", "word", or "char"')
        
        # Save the canvas to the save_path
        if save_path.endswith('png') or save_path.endswith('jpg'):
            cv2.imwrite(save_path, to_save)
        else:
            cv2.imwrite(os.path.join(save_path, str(img_idx) + '.png'), to_save)
