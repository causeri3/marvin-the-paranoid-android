from utils.yolo_utils.payloads import json_payload, image_payload
from utils.yolo_utils.object_detection import ObjectDetector

import onnxruntime as nxrun
import numpy as np
import logging
import time
import cv2


def load_model(model_path,
               providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
    logging.info("Detected and connected {} device with onnx library".format(nxrun.get_device()))
    session = nxrun.InferenceSession(model_path, providers=providers)
    logging.info("The model should be loaded with {} and expects input shape: {}".format(providers, session.get_inputs()[0].shape))
    return session


def predict(image,
            session,
            return_json=True,
            return_image=False):

    """Gives object detection for one image from yolo v7 via onnx runtime session
    :param image: Input image in bytes or as np array
    :param session: ONNX Runtime inference session via CPU or GPU depending on setting
    :param return_json: returning json payload with results (tags, boxes, confidences)
    :param return_image: Returning image with labels, confidences and boxes as bytes
    :returns If none of the above return options is chosen the image will open locally"""

    start_time = time.time()

    if len(image) == 0:
        raise logging.warning("The input appears to be empty")

    if type(image) != bytes and type(image) != np.ndarray:
        raise logging.warning("""
                            Media type has to be an image encoded in bytes or as numpy array,
                            however input has type {}""".format(type(image)))

    if type(image) == bytes:
        image_array = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image_array, flags=1)
        if image is None:
            logging.warning("Could not convert bytes to image array")

    if type(image) != np.ndarray:
        logging.warning("""
        Media type has to be an image encoded in bytes or as numpy array,
        however input has type {}""".format(type(image)))

    logging.debug("""
        The converted image has shape: {}
        """.format(image.shape))

    logging.debug("Invoking inference with provider setting: {}".format(session._providers))

    detected_objects = ObjectDetector(image, session).detect()

    logging.debug("Detected {} objects".format(len(detected_objects)))

    if return_json and return_image:
        json_output = json_payload(detected_objects)
        bytes_output = image_payload(detected_objects, image)
        end_time = time.time()
        logging.debug("One prediction took {:.2f} seconds".format(end_time - start_time))
        return bytes_output, json_output

    if return_json:
        json_output = json_payload(detected_objects)
        end_time = time.time()
        logging.debug("One prediction took {:.2f} seconds".format(end_time - start_time))
        return json_output

    if return_image:
        bytes_output = image_payload(detected_objects, image)
        end_time = time.time()
        logging.debug("One prediction took {:.2f} seconds".format(end_time - start_time))
        return bytes_output

    else:
        output_bytes = image_payload(detected_objects, image)
        output_array = np.frombuffer(output_bytes, dtype=np.uint8)
        output_image = cv2.imdecode(output_array, flags=1)
        end_time = time.time()
        logging.info("One prediction took {:.2f} seconds".format(end_time - start_time))
        cv2.imshow('image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

