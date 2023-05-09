from pathlib import Path
import logging
import json
import gradio as gr
import torch
import pyaudio
from functools import partial
import queue
import subprocess

from extensions.fast_tts.piper import Piper
from extensions.fast_tts.play_audio import play

from modules import shared

torch._C._jit_set_profiling_mode(False)

current_file_dir = Path(__file__).parent.absolute()
last_text = ''
is_playing = False
audio_stream = None
config = None

params = {
    # TODO Remember settings
    'activate': True,
    'save_audio': True,
    'available_models': [],
    'model': '',
    'model_config': '',
    # TODO Add speaker id
    'speaker_id': None,
    # TODO Add noise scale
    'noise_scale': 0.0,
    # TODO Add length scale
    'length_scale': 1.0,
    # TODO Add noise weight
    'noise_w': 0.0,
    'language': 'en',
    'sample_rate': 16000,
    'device': 'cpu',
}


def load_available_models():
    available_models = []

    # Read available models from the models folder
    for model_file in Path(f'{current_file_dir}/models').glob('*.onnx'):
        logging.info(f'Found model {model_file} in the models folder')
        available_models.append(model_file.absolute().__str__())
    if len(available_models) == 0:
        logging.warn('No models found in the models folder. Please download a compatible espeak model')
    params['available_models'] = available_models
    return available_models

def load_config_for_model(model: str):
    with open(f'{model}.json', 'r', encoding='utf8') as f:
        global config
        # Lossy decode the JSON file to a string
        config = json.load(f)

    return model + ".json"

current_params = params.copy()

def load_model():
    available_models = params['available_models']
    if len(available_models) == 0:
        logging.error('No models found in the models folder. Please download a compatible espeak model')
        available_models = load_available_models()
    if params['model'] == '' or params['model'] not in available_models:
        # Make first available model the default
        params['model'] = available_models[0]
        logging.info(f'No model selected. Selecting the first available model: {params["model"]}')
        params['model_config'] = load_config_for_model(params['model'])
    # TODO: Fix CUDA
    voice = Piper(params['model'], config_path=params['model_config'], use_cuda=params['device'] == 'gpu')
    synthesize = partial(
        voice.synthesize,
        speaker_id=params['speaker_id'],
        length_scale=params['length_scale'],
        noise_scale=params['noise_scale'],
        noise_w=params['noise_w'],
    )
    return synthesize

def state_modifier(state):
    state['stream'] = False
    return state

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    shared.processing_message = "*Is recording a voice message...*"
    return string


def output_stream(queue: queue.Queue, additionalParameters = None):
    """
    This function is applied to the output text stream of the model. The input is a channel
    """

    end_token: str = additionalParameters.endToken | "<#END#>"
    is_streaming = True
    while is_streaming:
        text = queue.get()
        # If the text ends with the end token, stop the stream
        if text.endswith(end_token):
            is_streaming = False
            text = text[:-len(end_token)]
        # 'espeak', '-v', 'en-us', '-s', '120', '-p', '70'
        subprocess.run(['espeak-ng, '-v', 'en-us', '-s', '120', '-p', '70',', text])


def output_modifier(text):
    """
    This function is applied to the model outputs.
    """

    return text
    global model, current_params, last_text

    original_text = text

    # if last_text == '' or last_text == text:
    #     # End of generation
    #     last_text = text
    #     return text
    # else:
    #     # Remove prefix from text
    #     text = text[len(last_text):]
    #     last_text = last_text + text
    logging.info(f"Generating audio for: {text}")

    # Check if the parameters have changed
    for i in params:
        if params[i] != current_params[i]:
            current_params = params.copy()
            break
    # Load model if it has changed or if it is the first time
    if model is None or params['model'] is not current_params['model']:
        model = load_model()

    if not params['activate']:
        return text

    if text == '':
        text = '*Empty reply, try regenerating*'
    else:
        output_dir = Path(f'{current_file_dir}/output')
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_bytes = model(text)

        global audio_stream
        if audio_stream is None:
            p = pyaudio.PyAudio()

            audio_stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=config['audio']['sample_rate'],
                            output=True,
                            )

        # if not audio_stream.is_playing():
        #     audio_to_play = b''.join(buffer)
        #     Thread(target=audio_stream.write, args=(audio_to_play,)).start()

        # TODO: Play audio stream instead of waiting for the whole audio to be generated (using pyAudio callback with a buffer)
        # Thread(target=audio_stream.write, args=(wav_bytes,)).start()
        audio_stream.write(wav_bytes)

        # if (params['save_audio']):
        #     # TODO: Better nameing of the audio files
        #     with open(f"{output_dir}/test.wav", "wb") as output_file:
        #         output_file.write(wav_bytes)
        # Play audio in another thread to allow for continuous streaming
        # global is_playing
        # if not is_playing:
        #     Thread(target=play, args=(f"{output_dir}/test.wav",)).start()
        #     is_playing = True

    logging.debug(text)
    # shared.processing_message = "*Is typing...*"
    return original_text


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """
    return string


def setup():
    global model
    model = load_model()
    load_available_models()

def update_model_selection(x):
    global available_models
    available_params = load_available_models()
    return gr.Dropdown.update(choices=available_params)

def ui():
    # Gradio elements
    with gr.Accordion("Fast TTS"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            save_audio = gr.Checkbox(value=params['save_audio'], label='Save audio to file')

        try:
            model = gr.Dropdown(value=params['model'], choices=[x['model'] for x in params['available_models']], label='TTS model')
        except:
            logging.warn('Error loading models.')
            model = gr.Dropdown(value='', choices=params['available_models'], label='TTS model')
        reload = gr.Button(label='Reload models')

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    save_audio.change(lambda x: params.update({"save_audio": x}), save_audio, None)
    reload.click(update_model_selection, model, None)
    model.change(lambda x: params.update({"model": x}), model, None)
