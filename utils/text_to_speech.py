from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import sounddevice as sd


def load_tts_model(model_name="facebook/fastspeech2-en-ljspeech"):
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        model_name,
        arg_overrides={"vocoder": "hifigan", "fp16": False})
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator([models[0]], cfg)
    return models[0], task, generator


model, task, gen = load_tts_model()


def inference_tts(text="Optimism is a nice expression for stubbornness", loaded_task=task, tts_model=model, generator=gen):
    sample = TTSHubInterface.get_model_input(loaded_task, text)
    wav, rate = TTSHubInterface.get_prediction(loaded_task, tts_model, generator, sample)
    sd.play(wav, rate)
