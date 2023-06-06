import gradio as gr
import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import time
from PIL import Image
import os
from datetime import datetime
import csv

# Initialize Object Detection
od = ObjectDetection()


def process_video(input_video):
    cap = cv2.VideoCapture(input_video)

    output_path = "output.mp4"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *"mp4v"), fps, (width, height))

    iterating, frame = cap.read()
    count = 0
    center_points = []

    while iterating:

        # flip frame vertically
        count += 1
        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w)/2)
            cy = int((y + y + h)/2)
            center_points.append((cx, cy))
            print("Frame NO: ", count, " ", x, y, w, h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        for pt in center_points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        output_video = gr.Video(label="output")

    process_video_btn.click(process_video, input_video, [
                            processed_frames, output_video])

demo.queue()
demo.launch()
