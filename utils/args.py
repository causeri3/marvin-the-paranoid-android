from argparse import ArgumentParser
from os.path import join


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-m',
                        '--model-path',
                        type=str,
                        required=False,
                        # default=join('', '', 'files', 'models', 'yolov7.onnx'),
                        default=join('', 'inference', '../files', 'models', 'yolov7.onnx'),
                        help='Path to YOLO Version 7 ONNX file')
    parser.add_argument('-c',
                        '--cam-device-number',
                        required=False,
                        default=None,
                        help='Overwrite camera device number with an integer')
    parser.add_argument('-ok',
                        '--openai-key',
                        required=True,
                        type=str,
                        help='Your open ai key to enable the API')
    parser.add_argument('-s',
                        '--see-detection',
                        required=False,
                        type=bool,
                        default=True,
                        help='See object detection in video output')
    parser.add_argument('-i',
                        '--interval',
                        required=False,
                        type=int,
                        default=45,
                        help='Seconds for which detected objects get detected until ChatGPT gets called')
    parser.add_argument('-vp',
                        '--video-path',
                        required=False,
                        default=None,
                        help='String containing path to video')
    parser.add_argument('-y',
                        '--youtube-url',
                        required=False,
                        default=None,
                        help='String containing url to youtube video')

    # to not get into trouble with uvicorn args
    args = parser.parse_known_args()
    return args