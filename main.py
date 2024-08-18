from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")
imagePath = 'img/9f9a0633144adbefc90b4a53f8bf335.jpg'

# Perform object detection on an image using the model
results = model(imagePath)

for result in results:
    im_array = result.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('result.jpg')