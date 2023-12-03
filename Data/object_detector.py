
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tqdm

image_dir = 'Datasets/M2E2/voa/m2e2_rawdata/image/image'
# all_images = [image_dir + '/' + f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

model = YOLO('yolov8l.pt')

detector_result = {}

index = 0

for image in os.listdir(image_dir):

    print(index)
    index+=1

    detector_result.setdefault(image,{})
    img = Image.open(image_dir + '/' + image) # from PIL
    results = model.predict(source=img)  # return a list
    # print(print(results[0].boxes))
    # print("###")
    # print(results[0].boxes.xyxy)
    # print(results[0].boxes.cls)
    # print(results[0].boxes.conf)
    # detector_result[image]['boxes'] = results[0].boxes
    detector_result[image]['xyxy'] = results[0].boxes.xyxy.cpu()
    detector_result[image]['cls'] = results[0].boxes.cls.cpu()
    detector_result[image]['conf'] = results[0].boxes.conf.cpu()
    # break
    res_plotted = results[0].plot()
    cv2.imwrite(os.path.join('Datasets/M2E2/voa/object_detection/yolov8l', image), res_plotted)


with open("Datasets/M2E2/voa/object_detection/yolov8l/bboxes.pkl", "wb") as f:
    pickle.dump(detector_result, f)

with open("Datasets/M2E2/voa/object_detection/yolov8l/bboxes.pkl", "rb") as f:
    detector_result = pickle.load(f)

print(len(detector_result))