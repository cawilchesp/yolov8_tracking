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


def draw_boxes(image: np.array, ds_output: np.array) -> None:
    """
    Draw bounding boxes on frame
    """
    for box in enumerate(ds_output):
        box_xyxy = box[1][0:4]
        object_id = box[1][-2]
        class_id = box[1][-1]
        if class_id in class_filter:
            # Draw box
            x1, y1, x2, y2 = [int(j) for j in box_xyxy]
            color = compute_color(class_id)
            label = f'{class_names[class_id]} {int(object_id)}'
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw label
            t_size = cv2.getTextSize(label, 0, 1/3, 1)[0]
            cv2.rectangle(image, (x1, y1-t_size[1]-3), (x1 + t_size[0], y1+3), color, -1, cv2.LINE_AA)
            cv2.putText(image, label, (x1, y1 - 2), 0, 1/3, [225, 255, 255], 1, cv2.LINE_AA)


def draw_trajectories(image: np.array, ds_output: np.array) -> None:
    """
    Draw object trajectories on frame
    """
    if (len(ds_output) > 0):
        objects_id = ds_output[:,-2]
    
        # Remove tracked point from buffer if object is lost
        for key in list(track_deque):
            if key not in objects_id:
                track_deque.pop(key)

    for box in enumerate(ds_output):
        box_xyxy = box[1][0:4]
        object_id = box[1][-2]
        class_id = box[1][-1]
        if class_id in class_filter:
            # Draw track line
            x1, y1, x2, y2 = [int(j) for j in box_xyxy]
            center = (int((x2+x1)/2), int((y2+y1)/2))
            if int(object_id) not in track_deque:
                track_deque[int(object_id)] = deque(maxlen=32)
            track_deque[int(object_id)].appendleft(center)
            color = compute_color(class_id)
            for point1, point2 in zip(list(track_deque[int(object_id)]), list(track_deque[int(object_id)])[1:]):
                cv2.line(image, point1, point2, color, 2, cv2.LINE_AA)


def write_csv(csv_path: str, ds_output: np.array, frame_number: int) -> None:
    """
    Write object detection results in csv file
    """
    for box in enumerate(ds_output):
        box_xyxy = box[1][0:4]
        object_id = box[1][-2]
        class_id = box[1][-1]
        if class_id in class_filter:
            x1, y1, x2, y2 = [int(j) for j in box_xyxy]

            # Save results in CSV
            with open(f'{csv_path}.csv', 'a') as f:
                f.write(f'{frame_number},{object_id},{class_names[class_id]},{x1},{y1},{x2-x1},{y2-y1}\n')








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
    model_file = 'yolov8m'
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

            # Non-Maximum Suppression
            values, indices = torch.sort(boxes.conf, descending=True)
            sorted_boxes = boxes[indices]

            nms_indices = torch.ops.torchvision.nms(sorted_boxes.xyxy, boxes.conf, 0.5)
            nms_boxes = sorted_boxes[nms_indices]

            # Deep SORT tracking
            ds_output = deepsort.update(nms_boxes.xywh.cpu(), nms_boxes.conf.cpu(), nms_boxes.cls.cpu(), image)
            
            draw_boxes(image, ds_output)
            draw_trajectories(image, ds_output)
            write_csv(output_file_name, ds_output, frame_number)

        vid_writer.write(image)
        # output_name = f'{output_folder_name}image_{str(frame_number).zfill(6)}.png'
        # cv2.imwrite(output_name, image)
        
        # Increase frame number
        frame_number += 1

        # Visualization
        cv2.imshow('source', image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break

    vid_writer.release()


if __name__ == "__main__":
    with torch.no_grad():
        main()
        