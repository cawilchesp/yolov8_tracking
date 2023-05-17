from ultralytics import YOLO

import cv2
import torch
import numpy as np

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from collections import deque

# For debugging
from icecream import ic


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

data_deque = {}

# class names
class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
             'hair drier', 'toothbrush' ]


def xyxy_to_xywh(xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    x1, y1, x2, y2 = xyxy
    bbox_left = min([x1 , x2 ])
    bbox_top = min([y1 , y2 ])
    bbox_w = abs(x1 - x2)
    bbox_h = abs(y1 - y2)
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color(label: int) -> tuple:
    """
    Adds color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color_table = {
        0: (85, 45, 255), # person
        1: (7, 127, 15),  # bicycle
        2: (255, 149, 0), # Car
        3: (0, 204, 255), # Motobike
        5: (0, 149, 255), # Bus
        7: (222, 82, 175) # truck
    }

    if label in color_table.keys():
        color = color_table[label]
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    
    return color


def draw_boxes(image: np.array, bbox: np.array, object_id: np.array, identities: np.array) -> None:  	                                                                                                                                                                                                            
    """
    Draw bounding boxes on frame
    """
    # Remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(j) for j in box]

        center = (int((x2+x1)/2), int((y2+y1)/2))

        id_num = int(identities[i]) if identities is not None else 0
        if id_num not in data_deque:
            data_deque[id_num] = deque(maxlen=32)

        data_deque[id_num].appendleft(center)

        color = compute_color(object_id[i])
        label = f'{class_names[object_id[i]]} {id_num}'

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        # Draw trajectory
        for k in range(1, len(data_deque[id_num])):
            if data_deque[id_num][k-1] is None or data_deque[id_num][k] is None:
                continue

            cv2.line(image, data_deque[id_num][k-1], data_deque[id_num][k], color, 2)

        # Draw labels
        t_size = cv2.getTextSize(label, 0, 1/3, 1)[0]
        cv2.rectangle(image, (x1, y1-t_size[1]-3), (x1 + t_size[0], y1+3), color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y1 - 2), 0, 1/3, [225, 255, 255], 1, cv2.LINE_AA)

        # # Save results in CSV
        # with open(csv_path, 'a') as f:
        #     f.write(f'{frame_num},{id_num},{str(names[object_id[i]])},{x1},{y1},{x2-x1},{y2-y1},0,\n')


def main():
    # Source

    # source_folder = 'D:/SIER/Videos/DEEP_CCTV/'
    # source_file = '20230412_CCTV_Barrio_Triste'

    source_folder = 'D:/SIER/Videos/Aforo_Bus/'
    source_file = 'sgtcootransvi.dyndns.org_01_2023051112113649'


    cap = cv2.VideoCapture(f'{source_folder}{source_file}.mp4')
    fourcc = 'mp4v'  # output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output
    output_folder = f'{source_folder}{source_file}/'
    vid_writer = cv2.VideoWriter(f'{output_folder}output_{source_file}.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))


    # Initialize Deep-SORT
    cfg_deep = get_config()
    cfg_deep.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize YOLOv8 Model
    model = YOLO('../weights/yolov8m.pt')

    # Class filter
    class_filter = [0,1,2,3,5,7]

    # Run inference
    frame_number = 0
    while True:
        success, image = cap.read()
        
        # Process detections
        detections = model(image, stream=True)
        
        for r in detections:
            boxes = r.boxes

            values, indices = torch.sort(boxes.conf, descending=True)
            sorted_boxes = boxes[indices]

            nms_indices = torch.ops.torchvision.nms(sorted_boxes.xyxy, boxes.conf, 0.5)
            nms_boxes = sorted_boxes[nms_indices]


            outputs = deepsort.update(nms_boxes.xywh.cpu(), nms_boxes.conf.cpu(), nms_boxes.cls.cpu(), image)
            
            if len(outputs) > 0:
                ic(outputs)
                # raise SystemExit(0)
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]

                draw_boxes(image, bbox_xyxy, object_id, identities)

                

        vid_writer.write(image)

        # output_name = f'{output_folder}image_{str(frame_number).zfill(6)}.png'
        # cv2.imwrite(output_name, image)
        # frame_number += 1

                

        cv2.imshow('source', image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break

    vid_writer.release()

if __name__ == "__main__":
    with torch.no_grad():
        main()
        