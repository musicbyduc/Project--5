from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO('/Xu ly anh/yolov8/runs/detect/train5/weights/best.pt')

# run inference on the source
results = model.track(source='VIDEO_1.mp4', show=True, tracker="bytetrack.yaml")

