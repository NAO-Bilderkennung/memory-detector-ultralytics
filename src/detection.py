import os.path
from time import perf_counter

import wx.py.path
from ultralytics import YOLO
import numpy as np
import cv2 as cv


def bb_tensor_to_lists(tensor):
    xyxy = tensor[0].tolist()  # BB is a (1, 4) Tensor, so we need to convert it to a Python list first

    top_left = (int(xyxy[0]), int(xyxy[1]))
    bottom_right = (int(xyxy[2]), int(xyxy[3]))

    return top_left, bottom_right


class ObjectDetector:
    def __init__(self, model_type, cuda_device=None, conf_thresh=0.5, font=cv.FONT_HERSHEY_SIMPLEX):
        if not os.path.exists("models"):
            os.mkdir("models")

        self.model_type = model_type
        self.model = YOLO(f"models/{self.model_type}.pt")
        self.device = cuda_device if cuda_device is not None else "cpu"
        self.conf_thresh = conf_thresh

        class_count = len(self.model.names)
        self.colors = np.random.rand(class_count, 3) * 255
        self.font = font

        self.should_filter = True

        self.results = None
        self.last_image = None
        self.result_image = None
        self.prediction_time = None
        self.box_draw_time = None

    def detect(self, image):
        starting_time = perf_counter()
        self.results = self.model.predict(source=image, device=self.device)
        prediction_time = perf_counter()

        result_image = image.copy()
        for result in self.results:
            for box in result.boxes:
                if self.should_filter and (box.conf.item() <= self.conf_thresh):
                    continue

                self._draw_box(result_image, box)

        box_draw_time = perf_counter()

        self.last_image = image.copy()
        self.result_image = result_image

        self.prediction_time = prediction_time - starting_time
        self.box_draw_time = box_draw_time - prediction_time

    def get_class_name_of_box(self, box):
        class_index = int(box.cls.item())
        return self.model.names[class_index]

    def get_color_of_box(self, box):
        class_index = int(box.cls.item())
        return self.colors[class_index]

    def _draw_box(self, target, result):
        top_left, bottom_right = bb_tensor_to_lists(result.xyxy)

        item_class = self.get_class_name_of_box(result)
        color = self.get_color_of_box(result)

        text_bottom_left = (top_left[0], top_left[1] - 5)  # Offset the text to the top for better legibility

        cv.rectangle(target, top_left, bottom_right, color, 2)
        cv.putText(target, item_class, text_bottom_left, self.font, 1, color, 2, cv.LINE_AA)
