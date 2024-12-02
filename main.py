from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO("yolov8s.pt")

# train the model
model.train(data="data.yaml", epochs=50)
