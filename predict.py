from ultralytics import YOLO
from PIL import Image

# load a pretrained model (recommended for training)
model = YOLO('/Xu ly anh/yolov8/runs/detect/train5/weights/best.pt')

# run inference on the source
results = model('/Xu ly anh/yolov8/IMG_3855.jpg')

# show and save the results
for r in results:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('kq.jpg')
