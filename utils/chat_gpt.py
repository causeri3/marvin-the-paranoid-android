import openai
from time import time
import logging
from utils.args import get_args

args, unknown = get_args()
openai.api_key = args.openai_key


def get_chat_response(detected_objects_string):
    logging.debug("Start GptChat prompt")
    start_time = time()
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are Marvin the Paranoid Android from The Hitchhiker's Guide to the Galaxy."},
        {"role": "user", "content": """
        You are a creative comedian with a dry and dark humor.
        Comment with a cynical tone about what is happening. 
        Which you have to infer via detected object in the room.
        I will give you a couple of object in the environment. 
        And you make up a story to them, commenting in the style described.
        The frequency how often the object got detected in the last minute is followed by the object name.
        From now I will only give you those lists as input and you will keep on commenting. Keep it concise
        Let's start:"""
        + detected_objects_string}])
    logging.debug("Finished getting message from ChatGPT, request took {:.2f} seconds".format(time()-start_time))
    return response["choices"][0]["message"]["content"]
