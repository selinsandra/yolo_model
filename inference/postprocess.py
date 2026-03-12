# inference/postprocess.py

def filter_by_class(detections, allowed_classes):
    """
    Filters detections by allowed class names.

    :param detections: List[Dict]
    :param allowed_classes: List[str]
    """
    return [
        det for det in detections
        if det["class_name"] in allowed_classes
    ]