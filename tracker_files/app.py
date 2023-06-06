from deep_sort import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
import time
from yolov3.configs import *
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
import tensorflow as tf
import numpy as np
import cv2
import gradio as gr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


max_cosine_distance = 0.7
nn_budget = None

# initialize deep sort object
Yolo = Load_Yolo_model()
model_filename = 'model_data/mars-small128.pb'
CLASSES = YOLO_COCO_CLASSES
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
input_size = 416
score_threshold = 0.3
iou_threshold = 0.45
Track_only = ["bicycle", "car", "motorbike",
              "bus", "truck"]
show = False


# Line (Lane)
Line1 = [(762, 480), (177, 1071)]
Line2 = [(900, 474), (951, 1071)]
Line3 = [(1043, 462), (1826, 1070)]

# Area
area1 = [(762, 480), (177, 1071), (951, 1071), (900, 474)]
area2 = [(900, 474), (951, 1071), (1826, 1070), (1043, 462)]

# Line for capture Images 
cap_Line = [(483, 774), (1410, 774)]


def process_video(input_video):
    times, times_2 = [], []
    cap = cv2.VideoCapture(input_video)

    output_path = "output.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    # output_path must be .mp4
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *"mp4v"), fps, (width, height))

    iterating, frame = cap.read()

    while iterating:

        # track
        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        image_data = image_preprocess(np.copy(original_frame), [
                                      input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(
            pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')
        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) != 0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(
                    int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)
        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id  # Get the ID for the particular track
            # Get predicted object index by object name
            index = key_list[val_list.index(class_name)]
            # Structure data, that we could use it with our draw_bbox function
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index])
            print(bbox)
            coor = np.array(bbox[:4], dtype=np.int32)
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
            cx = int((x1 + x2)/2)
            cy = int((y1+y2)/2)
            # cv2.circle(original_frame, (cx,cy), 5, (0, 0, 255), -1)
            # draw detection on frame
        
        image = draw_bbox(original_frame, tracked_bboxes,
                          CLASSES=CLASSES, tracking=True)

        

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        image = cv2.putText(image, "Time: {:.1f} FPS".format(
            fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        for line in [Line1, Line2, Line3]:
            cv2.line(image, line[0], line[1], (255, 0, 0), 3)
        
        cv2.line(image, cap_Line[0], cap_Line[1], (0, 0, 255), 2)
        
        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(
            ms, fps, fps2))
        # if output_path != '': out.write(image)
        # if show:
        #     frame = cv2.imshow('output', image)

        # display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        video.write(frame)
        yield display_frame, None

        iterating, frame = cap.read()

    video.release()
    yield display_frame, output_path


with gr.Blocks() as demo:
    with gr.Row():
        input_video = gr.Video(label="input")
        examples = gr.Examples(["video1.mp4"], inputs=input_video)
        process_video_btn = gr.Button("process video")

    with gr.Row():

        processed_frames = gr.Image(label="last frame")

    with gr.Row():
        output_video = gr.Video(label="output")

    process_video_btn.click(process_video, input_video, [
                            processed_frames, output_video])

demo.queue()
demo.launch()
