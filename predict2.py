import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO('/Xu ly anh/yolov8/runs/detect/train5/weights/best.pt')

# run inference on the source
results = model.track(source=0, show=True, tracker="bytetrack.yaml")
