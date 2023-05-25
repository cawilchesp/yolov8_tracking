from ultralytics import YOLO

import cv2
import torch
import numpy as np

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from collections import deque

# For debugging
from icecream import ic

# VARIABLES

# object tracks
track_deque = {}

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

# class filter
class_filter = [0,1,2,3,5,7]
# class_filter = [0]

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


def draw_masks(image: np.array, boxes: torch.tensor, masks: torch.tensor) -> None:
    for index, mask in enumerate(masks):
        class_id = int(boxes.cls[index])

        if class_id in class_filter:
            object_mask = mask.data.cpu().numpy()[0]
            color = np.array(compute_color(class_id), dtype='uint8')
            masked_img = np.where(object_mask[...,None], color, image)
            image = cv2.addWeighted(image, 0.6, masked_img, 0.4, 0)

    return image
        

def main():
    # Initialize Deep-SORT
    cfg_deep = get_config()
    cfg_deep.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize YOLOv8 Model
    model_folder = '../weights/'
    model_file = 'yolov8m-seg'
    model = YOLO(f'{model_folder}{model_file}.pt')

    # Source
    # source_folder_name = 'D:/SIER/Videos/DEEP_CCTV/'
    # source_file_name = '20230412_CCTV_Barrio_Triste'

    # source_folder_name = 'D:/SIER/Videos/Aforo_Bus/'
    # source_file_name = 'sgtcootransvi.dyndns.org_01_2023051112113649'
    # source_file_name = 'sgtcootransvi.dyndns.org_01_20230511120030951'
    # source_file_name = 'sgtcootransvi.dyndns.org_01_20230511120254332'
    # source_file_name = 'sgtcootransvi.dyndns.org_01_20230511121459931'

    source_folder_name = 'D:/SIER/Videos/Ruta_Costera/'
    source_file_name = 'PTZ010'
    # source_file_name = 'CAR021'
    # source_file_name = 'OP030'


    cap = cv2.VideoCapture(f'{source_folder_name}{source_file_name}.avi')
    if not cap.isOpened():
        raise RuntimeError('Cannot open source')
    

    print('***             Video Opened             ***')

        
    fourcc = 'mp4v'  # output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output
    output_folder_name = f'{source_folder_name}{source_file_name}/'
    output_file_name = f'{output_folder_name}output_{source_file_name}_{model_file}'
    vid_writer = cv2.VideoWriter(f'{output_file_name}.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    
    # Run YOLOv8 inference
    print('***             Video Processing Start             ***')
    frame_number = 0
    while True:
        print(f'Progress: {frame_number}/{frame_count}')

        success, image = cap.read()
        if not success: break
        
        # Process YOLOv8 detections
        detections = model(image, stream=True)

        for r in detections:
            boxes = r.boxes
            masks = r.masks

            segmented_image = draw_masks(image, boxes, masks)

        # Visualization
        cv2.imshow('source', segmented_image)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break
                



if __name__ == "__main__":
    with torch.no_grad():
        main()
        