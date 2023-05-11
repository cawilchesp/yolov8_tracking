from ultralytics import YOLO
import cv2
import numpy as np


data_deque = {}
frame_deque = {}


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


def draw_boxes(img: np.array, bbox: tuple, class_number: int, confidence: float) -> None:
    """
    Draw bounding boxes on frame
    """
    x1, y1, x2, y2 = bbox
    
    center = (int((x2+x1)/2), int((y2+y1)/2))

    class_filter = [0,1,2,3,5,7]
    if cls in class_filter:
        color = compute_color(class_number)

        # Draw boxes
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Draw centers
        cv2.circle(img, center, 2, color, 2, cv2.LINE_AA)

        # Draw labels
        label = f'{class_names[class_number]} {confidence:.2f}'
        t_size = cv2.getTextSize(label, 0, 1/3, 1)[0]
        cv2.rectangle(img, (x1, y1-t_size[1]-3), (x1 + t_size[0], y1+3), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y1 - 2), 0, 1/3, [225, 255, 255], 1, cv2.LINE_AA)




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


model = YOLO('../weights/yolov8x.pt')

source_file = 'D:/SIER/Datasets/inside_bus/train/images/image_12.jpg'

img = cv2.imread(source_file)




results = model(img, stream=True)
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        bbox = (int(x1), int(y1), int(x2), int(y2))

        # Confidence
        conf = box.conf[0]

        # Class name
        cls = int(box.cls[0])

        draw_boxes(img, bbox, cls, conf)

            




    cv2.imshow('source', img)
    cv2.waitKey(0)
