from ultralytics import YOLO
import torch


class YOLOModel:
    def __init__(self, model_name: str, device: str):
        self.device = self._select_device(device)
        self.model = YOLO(model_name)

    def _select_device(self, device: str):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def predict(self, image, img_size: int, conf: float, iou: float):
        return self.model.predict(
            source=image,
            imgsz=img_size,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )