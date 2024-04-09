from utils.args import get_args

from ultralytics import YOLO
import os
import logging
import shutil


def get_model():
    args, unknown = get_args()

    if args.model_size == 'medium':
        yolo = "YOLOv8m"
    elif args.model_size == 'large':
        yolo = "yolov8l"
    else:
        yolo = "yolov8s"
    onnx_model_path = os.path.join(args.model_path, yolo + '.onnx')
    if not os.path.exists(onnx_model_path):
        logging.info('Model {} not found, will be downloaded.'.format(yolo + '.onnx'))
        model = YOLO(yolo)
        logging.info('Model  {} got downloaded, will be converted to onnx format.'.format(yolo + '.pt'))
        model_path = model.export(format="onnx", imgsz=[640, 640])
        os.remove(yolo + '.pt')
        if os.path.exists(model_path):
            shutil.move(model_path, onnx_model_path)
            logging.info('Model {} successfully downloaded, converted and moved to {}.'.format(model_path, onnx_model_path))
    return onnx_model_path

