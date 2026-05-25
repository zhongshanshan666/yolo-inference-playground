# Import registers all model classes with the registry.
from yolo_playground.models.yolo import Yolo11Model, Yolo26Model, YoloV8Model
from yolo_playground.models.yolov5 import YoloV5Model

__all__ = ["YoloV5Model", "YoloV8Model", "Yolo11Model", "Yolo26Model"]
