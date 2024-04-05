from utils.text_to_speech import inference_tts
from utils.chat_gpt import get_chat_response

import cv2
import logging
import numpy as np
from time import time
import json
import threading
import os
from cap_from_youtube import cap_from_youtube


def return_camera_indexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
            logging.debug("Found device under number {}.".format(index))
        index += 1
        i -= 1
    return arr


def choose_device(device_numbers: list):
    if len(device_numbers) < 2:
        return device_numbers[0]
    # default for my preferred set-up (no deeper meaning)
    elif len(device_numbers) == 3:
        return device_numbers[1]
    else:
        return device_numbers[-1]


def count_n_sort_objects(object_list):
    u, count = np.unique(object_list, return_counts=True)
    count_sort_ind = np.argsort(-count)
    sorted_obj = u[count_sort_ind]
    sorted_count = count[count_sort_ind]
    list_of_counted_objects = [str(sorted_count[index]) + ' ' + sorted_obj[index] for index in range(len(count))]
    return list_of_counted_objects


def predict_n_stream(frame, predict_function, model_session, see_detection=True):

    if see_detection:
        combined_image_bytes, json_payload = predict_function(
            frame,
            model_session,
            return_image=True,
            return_json=True)

        combined_image_array = np.frombuffer(combined_image_bytes, dtype=np.uint8)
        combined_img = cv2.imdecode(combined_image_array, flags=1)
        return combined_img, json_payload

    else:
        json_payload = predict_function(
            frame,
            model_session,
            return_image=False,
            return_json=True)
        return frame, json_payload


def response_thread(text):
    response = get_chat_response(text)
    logging.info("ChatGPT response: " + response)
    inference_tts(response)


def draw_boxes(
        device_numbers: list,
        predict_function,
        model_session,
        see_detection=True,
        comment_interval=65,
        video_path=None,
        youtube_url=None):

    if video_path:
        video_name = os.path.basename(video_path)
        logging.info("Video {} gets loaded".format(video_name))
        cap = cv2.VideoCapture(video_path)
        window_name = "Video {}".format(video_name)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    elif youtube_url:
        cap = cap_from_youtube(youtube_url, '240p')
        window_name = "Youtube video of link:{}".format(youtube_url)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    else:
        device_numbers = choose_device(device_numbers)
        # Initialize the webcam
        cap = cv2.VideoCapture(device_numbers)
        window_name = "Your camera, device no: {}".format(device_numbers)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    start_time = time()
    object_list = []

    while cap.isOpened():

        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        image, json_payload = predict_n_stream(frame, predict_function, model_session, see_detection=see_detection)
        cv2.imshow(window_name, image)

        logging.debug(json.loads(json_payload)["objects"])
        if time()-start_time < comment_interval:
            object_list = object_list + json.loads(json_payload)["objects"]
        else:
            logging.info("List of all objects: " + " ".join(object_list))
            list_of_counted_objects = count_n_sort_objects(object_list)
            logging.info("List of all counted objects: " + " ".join(list_of_counted_objects))
            # run async
            threading.Thread(target=response_thread, args=(" ".join(list_of_counted_objects),)).start()
            object_list = []
            start_time = time()

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
