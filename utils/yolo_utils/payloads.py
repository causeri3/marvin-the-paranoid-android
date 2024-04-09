from utils.yolo_utils.labels import COCOLabels
from utils.yolo_utils.render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS

import logging
import cv2
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_payload(detected_objects):
    log_list = ['LABEL', 'CONFIDENCE']
    tags_list = []
    object_list = []

    for box in detected_objects:
        label = COCOLabels(box.class_id).name.title()
        log_list.append(label)
        log_list.append("{:.2f}".format(box.confidence))
        object_list.append(label)
        tag = {
            "label": label,
            "score": str(box.confidence),
            "box": {
                "x": box.x1,
                "y": box.y1,
                "width": box.y2 - box.y1,
                "height": box.x2 - box.x1,
            },
            "tag_id": box.class_id
        }
        tags_list.append(tag)

    log_str_format = ' \n {:<40} {:<40}' * (len(detected_objects) + 1)
    logging.info(log_str_format.format(*log_list))

    dict = {
        "tags": tags_list,
        "objects": object_list
    }

    return json.dumps(dict, cls=NpEncoder)


def image_payload(detected_objects, image):
    """
    :param detected_objects:
    :param image: numpy array with three dimensions (height, width, channels)
    :return: images with bounding boxes rawn on as bytes
    """
    log_list = ['LABEL', 'CONFIDENCE']
    output_image = image.copy()

    for box in detected_objects:
        label = COCOLabels(box.class_id).name.title()
        confidence = "{:.2f}".format(box.confidence)
        tag_text = label + ": " + confidence
        log_list.append(label)
        log_list.append(confidence)

        output_image = render_box(output_image, box.box(), color=tuple(RAND_COLORS[box.class_id % 64].tolist()))
        size = get_text_size(output_image, tag_text,
                             normalised_scaling=0.6)
        output_image = render_filled_box(output_image, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]),
                                         color=(220, 220, 220))
        output_image = render_text(output_image, tag_text,
                                   (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

    success, encoded_image = cv2.imencode('.jpg', output_image)

    if not success:
        logging.error("""
        Could not encode image in order to convert to bytes {}. 
        """. format(type(output_image)))

    image_bytes_output = encoded_image.tobytes()

    log_str_format = ' \n {:<40} {:<40}' * (len(detected_objects) + 1)
    logging.info(log_str_format.format(*log_list))

    return image_bytes_output


