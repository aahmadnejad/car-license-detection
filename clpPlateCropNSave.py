import os

import torch
import numpy as np
import cv2
from copy import deepcopy

from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box, plot_one_box_PIL

import clpSetting

if clpSetting.cuda and torch.cuda.is_available():
    device = select_device('cuda')
else:
    device = select_device('cpu')

model = attempt_load(clpSetting.YOLO_CLPD_WEIGHTS, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(clpSetting.image_size, s=stride)

if clpSetting.trace:
    model = TracedModel(model, device, clpSetting.image_size)

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))


def platedetection(source_image):

    img = letterbox(source_image, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=True)[0]

    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

    all_detections = []
    det_confidences = []

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()

            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                all_detections.append(coords)
                det_confidences.append(conf.item())

    return all_detections, det_confidences

def crop_plate(image, coord):
    cropped = image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
    return cropped

def get_and_save(input):
    if input is None:
        return None
    plate_detections, det_confidences = platedetection(input)
    plates = []
    detected_image = deepcopy(input)
    for coords in plate_detections:
        plate_region = crop_plate(input, coords)
        detected_image = plot_one_box_PIL(coords, detected_image, color=[0, 150, 255], line_thickness=2)
        rsize = cv2.resize(plate_region, (256,64), interpolation=cv2.INTER_LANCZOS4)
        plates.append(rsize)
    return plates

for fname in os.listdir(clpSetting.CARS_DATASET):
    img = get_and_save(cv2.imread(os.path.join(clpSetting.CARS_DATASET, fname)))
    for index,item in enumerate(img):
        cv2.imwrite(os.path.join(clpSetting.PLATEFRAME_SAVE_PATH, f'{index}-{fname}'), item)