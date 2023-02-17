import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


from utils.plots import plot_one_box, plot_one_box_PIL


import clpSetting
import clPlateDetection


def get_plates_from_webcam():

    source = cv2.VideoCapture(0)
    tracker = DeepSort(embedder_gpu=clpSetting.cuda)
    total_obj =0
    preds=[]
    while source.isOpened():
        ret, frame = source.read()
        # plate_detections =  detect_single_frame(frame)
        detections , confidences = clPlateDetection.platedetection(frame)
        detections = list(map(lambda bbox: clPlateDetection.pascal_voc_to_coco(bbox), detections))
        print(detections)
        if not ret:
            break

        if len(detections)  >  0:
            allDetections = [(detection, confidence, 'car-license-plate') for detection, confidence in zip(detections, confidences)]
            tracks = tracker.update_tracks(allDetections, frame=frame)
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                # Changing track bbox to top left, bottom right coordinates
                bbox = [int(position) for position in list(track.to_tlbr())]

                for i in range(len(bbox)):
                    if bbox[i] < 0:
                        bbox[i] = 0

                output_frame = {'track_id': track.track_id}

                # Appending track_id to list only if it does not exist in the list
                # else looking for the current track in the list and updating the highest confidence of it.
                if track.track_id not in list(set(pred['track_id'] for pred in preds)):
                    total_obj += 1
                    preds.append(output_frame)

                # Plotting the prediction.
                plotdetection = plot_one_box_PIL(bbox, frame, label=f'{str(track.track_id)}.', color=[255, 150, 0],
                                         line_thickness=3)

                if cv2.waitKey(1) == ord('q'):
                    break

                cv2.imshow('Main', plotdetection)
        else:
            if cv2.waitKey(1) == ord('q'):
                break
            cv2.imshow('Main', frame)

    source.release()