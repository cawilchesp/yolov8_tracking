from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

source_folder = 'D:/SIER/Videos/Aforo_Bus/'
# source_file = 'sgtcootransvi.dyndns.org_01_2023051112113649'
# source_file = 'sgtcootransvi.dyndns.org_01_20230511120030951'
# source_file = 'sgtcootransvi.dyndns.org_01_20230511120254332'
source_file = 'sgtcootransvi.dyndns.org_01_20230511121459931'

source = f'{source_folder}{source_file}.mp4'
model.predict(source=source, show=True)