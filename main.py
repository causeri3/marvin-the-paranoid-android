from utils.yolo_utils.predict import predict, load_model
from utils import video
import logging
from utils.args import get_args


args, unknown = get_args()

computing_providers = ['CUDAExecutionProvider']

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


if args.cam_device_number:
    logging.info("""Checking for camera devices ...""")
    available_devices = [args.cam_device_number]
    logging.info("""You got {} available devices which are presumed to be cameras.
                    You can address them under their numbers {}
                    If more than one available default is number 1 which is my external webcam.
                    """.format(len(available_devices), available_devices))

elif args.video_path:
    available_devices = []
else:
    available_devices = video.return_camera_indexes()

session_gpu = load_model(args.model_path, providers=computing_providers)

if __name__ == "__main__":
    video.draw_boxes(
        available_devices,
        predict,
        session_gpu,
        comment_interval=args.interval,
        see_detection=args.see_detection,
        video_path=args.video_path,
        youtube_url=args.youtube_url)
