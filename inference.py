from ultralytics import YOLO
import cv2

path = "/media/jackson/Extra/ID_Card_Project/dataset/images/z4484689715733_256c65d3707e500e732dda87210ca2e8.jpg"
model = YOLO(model='/media/jackson/Extra/ID_Card_Project/runs/detect/train12/weights/best.onnx')
results = model(path)
# boxes = results.boxes.xywh  # Boxes object for bbox outputs
# masks = results.masks  # Masks object for segmentation masks outputs
# keypoints = results.keypoints  # Keypoints object for pose outputs
# probs = results.probs  # Class probabilities for classification outputs

# print(f"box: {boxes}, name: {probs}")

boxes = results[0].boxes
# print(boxes.xyxy)
# print(boxes.cls)
img = cv2.imread(path)
for box,cls in zip(boxes.xyxy, boxes.cls):
    x1,y1,x2,y2 = box.cpu().numpy()
    img_ = img[int(y1):int(y2), int(x1):int(x2)]
    cv2.imshow(f"{cls}", img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

