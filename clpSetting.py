import os

# for cropping lincense plates from pld dataset to train an OCR model:
BASE_FOLDER = 'F:\\Datasets\\Car-LicencePlate'
CARS_DATASET = os.path.join(BASE_FOLDER,'images')
PLATEFRAME_SAVE_PATH ='C:\\Users\\aahr1\\Desktop\\New folder'

# Detection on local files:
LOCAL_DATA = './localData'

YOLO_CLPD_WEIGHTS = 'Yolo-clpd-best.pt'
cuda = True
image_size = 640
trace = True
confidence_threshold = 0.6
