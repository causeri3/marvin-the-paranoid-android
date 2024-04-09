from argparse import ArgumentParser
from os.path import join


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-m',
                        '--model-path',
                        type=str,
                        required=False,
                        default=join('', 'files', 'yolo-model'),
                        help='Path to YOLO Version 8 ONNX file')
    parser.add_argument('-cam',
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
                        action='store_false',
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
    parser.add_argument('-ms',
                        '--model-size',
                        required=False,
                        default='small',
                        help="""String containing 'medium' or 'large', the small model being  default""")
    parser.add_argument('-c',
                        '--confidence-threshold',
                        required=False,
                        default=0.25,
                        help="""Confidence threshold for detected object""")
    parser.add_argument('-iou',
                        '--iou-threshold',
                        required=False,
                        default=0.25,
                        help="""Threshold for Intersection over Union (IoU)""")


    # to not get into trouble with uvicorn args
    args = parser.parse_known_args()
    return args