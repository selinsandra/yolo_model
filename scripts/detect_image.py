import yaml
import cv2

from models.yolo_model import YOLOModel
from inference.detector import Detector
from inference.postprocess import filter_by_class
from utils.image_utils import load_image
from utils.visualization import draw_boxes
from utils.logger import get_logger


logger = get_logger(__name__)


def load_config():
    with open("configs/model.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    logger.info("Loading YOLO model...")
    model = YOLOModel(
        model_name=config["model_name"],
        device=config["device"]
    )

    detector = Detector(model, config)

    image = load_image("data/images/sample.jpg")

    logger.info("Running detection...")
    detections = detector.detect(image)

    detections = filter_by_class(detections, config["classes"])

    output = draw_boxes(image, detections)

    cv2.imwrite("outputs/images/result.jpg", output)
    logger.info("Detection complete. Saved output image.")


if __name__ == "__main__":
    main()