from typing import List, Dict


class Detector:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def detect(self, image) -> List[Dict]:
        results = self.model.predict(
            image=image,
            img_size=self.config["img_size"],
            conf=self.config["confidence_threshold"],
            iou=self.config["iou_threshold"],
        )

        detections = []

        for result in results:
            for box in result.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),   # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])]
                })

        return detections