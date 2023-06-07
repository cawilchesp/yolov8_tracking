from ultralytics import YOLO

import cv2
import torch
import numpy as np
import yaml
import time
from collections import deque

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from set_color import set_color

# For debugging
from icecream import ic


def time_synchronized():
    """
    PyTorch accurate time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def draw_boxes(image: np.array, object: np.array) -> None:
    """
    Draw bounding boxes on frame
    """
    x1, y1, x2, y2, _, class_id = object
    color = set_color(class_id)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_label(image: np.array, object: np.array) -> None:
    """
    Draw object label on frame
    """
    x1, y1, _, _, object_id, class_id = object
    color = set_color(class_id)
    label = f'{class_names[class_id]} {int(object_id)}'

    # Draw label
    t_size = cv2.getTextSize(label, 0, 1/3, 1)[0]
    cv2.rectangle(image, (x1, y1-t_size[1]-3), (x1 + t_size[0], y1+3), color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (x1, y1 - 2), 0, 1/3, [255,255,255], 1, cv2.LINE_AA)


def draw_trajectories(image: np.array, object: np.array) -> None:
    """
    Draw object trajectories on frame
    """
    x1, y1, x2, y2, object_id, class_id = object

    # Draw track line
    center = (int((x2+x1)/2), int((y2+y1)/2))
    if int(object_id) not in track_deque:
        track_deque[int(object_id)] = deque(maxlen=32)
    track_deque[int(object_id)].appendleft(center)
    color = set_color(class_id)
    for point1, point2 in zip(list(track_deque[int(object_id)]), list(track_deque[int(object_id)])[1:]):
        cv2.line(image, point1, point2, color, 2, cv2.LINE_AA)


def write_csv(csv_path: str, object: np.array, frame_number: int) -> None:
    """
    Write object detection results in csv file
    """
    x1, y1, x2, y2, object_id, class_id = object

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
    yolo_config = detection_config['YOLO']
    model = YOLO(f"weights/{yolo_config['YOLO_WEIGHTS']}.pt")

    # Initialize Input
    input_config = detection_config['INPUT']

    cap = cv2.VideoCapture(f"{input_config['FOLDER']}{input_config['FILE']}.avi")
    if not cap.isOpened():
        raise RuntimeError('Cannot open source')
    
    print('***                  Video Opened                  ***')
    
    fourcc = 'mp4v'  # output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output
    output_file_name = f"{input_config['FOLDER']}{input_config['FILE']}/output_{input_config['FILE']}_{yolo_config['YOLO_WEIGHTS']}_tracking"
    
    video_writer_flag = False
    if detection_config['SAVE']['VIDEO']:
        video_writer_flag = True
        video_writer = cv2.VideoWriter(f'{output_file_name}.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    
    # Run YOLOv8 inference
    print('***             Video Processing Start             ***')
    frame_number = 0
    
    while True:
        success, image = cap.read()
        if not success: break
        
        # Process YOLOv8 detections
        t1 = time_synchronized()
        detections = model.predict(
            source=image,
            conf=0.5,
            device=0,
            agnostic_nms=True,
            classes=detection_config['DETECTION']['CLASS_FILTER']
            )
        t2 = time_synchronized()

        # Deep SORT tracking
        deepsort_output = deepsort.update(detections[0].boxes.xywh.cpu(), detections[0].boxes.conf.cpu(), detections[0].boxes.cls.cpu(), image)

        for key in list(track_deque):
            if key not in deepsort_output[:,-2]:
                track_deque.pop(key)

        for object in deepsort_output:
            # Visualization
            if detection_config['SHOW']['BOXES']: draw_boxes(image, object)
            if detection_config['SHOW']['LABELS']: draw_label(image, object)
            if detection_config['SHOW']['TRACKS']: draw_trajectories(image, object)
            
            # Save in CSV
            if detection_config['SAVE']['CSV']: write_csv(output_file_name, object, frame_number)

        # Save video
        if video_writer_flag: video_writer.write(image)
        
        # Increase frame number
        print(f'Progress: {frame_number}/{frame_count}, Inference time: {1000*(t2-t1):.2f} ms')
        frame_number += 1

        # Visualization
        cv2.imshow('source', image)
        
        # Stop if Esc key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if video_writer_flag:
        video_writer.release()


if __name__ == "__main__":
    # Initialize Configuration File
    with open('detection_config.yaml', 'r') as file:
        detection_config = yaml.safe_load(file)

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

    # class_names = ['casco','chaqueta']

    with torch.no_grad():
        main()
        