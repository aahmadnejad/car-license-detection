from copy import deepcopy
import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
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

def pascal_voc_to_coco(x1y1x2y2):
    x1, y1, x2, y2 = x1y1x2y2
    return [x1, y1, x2 - x1, y2 - y1]