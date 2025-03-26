# Marvin the Paranoid Robot

## Description
From Video input to audio output. 
Via object detection - (yolov8, onnx format), LLM - (chatGPT, via API) and text-to-speech (fastspeech2-en-ljspeech).
One can use webcam, movie files or youtube videos as input. Compatible with Mac and Windows and properly Linux.

https://github.com/causeri3/marvin-the-paranoid-android/assets/67895591/5ffb696e-d7bc-430a-ae05-066325d817ec 

## Dependencies
### Python

`python==3.9`

### GPU

If you can leverage your GPU by having all CUDA dependencies installed, you can substitute `onnxruntime` with `onnxrunntime-gpu` in *requirements.txt*

**Got it running with:**

* [NVIDIA CUDA Driver](https://developer.nvidia.com/cuda-toolkit-archive) Version 11.5
* [CuDNN library](https://developer.nvidia.com/rdp/cudnn-archive) Version 8.3.0
* For Windows: [Microsoft Visual C++ (MSVC) compiler](https://visualstudio.microsoft.com/de/vs/community/)


### Python packages 
You can install them via

`pip install -r requirements.txt`

or even better if you use uv:
```sh
uv venv --python 3.9 
uv pip install -r requirements.txt
```


## Usage
You need an OpenAI Token to get it running
* webcam: `python main.py -ok <your key>`
* local video: `python main.py -ok <your key> -vp "path/to/your/video.mov"`
* youtube: `python main.py -ok <your key> -y "https://www.youtube.com/watch?v=uhkdUdXTUuc"`

## Args
See all arguments : `python yolo-chat-tts/main.py --help`

You can 
* choose between multiple camera devices
* pick the interval between the cynical comments
* choose whether the object detection is in your video or just in the logs
* choose a threshold for confidence 
* choose a threshold for IoU
* choose the model size

---


Thanks [Tien Luong Ngoc](https://github.com/tienluongngoc/yolov5_triton_inference_server/tree/main/clients/yolov5) & [Ibai Gorordo](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection), I took a bunch of useful code from your linked repositories
