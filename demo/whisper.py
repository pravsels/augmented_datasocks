import os
import time
import numpy as np
import sounddevice as sd
from pynput import keyboard
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3.5"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

def record_while_space_pressed(fs=16000):
    # print("Spacebar pressed: recording... (release spacebar to stop)")
    audio_blocks = []
    is_recording = True
    
    # Flag to track if space is pressed
    space_pressed = False
    
    # Define the key listeners
    def on_press(key):
        nonlocal space_pressed
        if key == keyboard.Key.space:
            space_pressed = True
            
    def on_release(key):
        nonlocal space_pressed, is_recording
        if key == keyboard.Key.space:
            space_pressed = False
            is_recording = False
            return False  # Stop the listener
    
    # Define a callback that appends each audio block to the list.
    def callback(indata, frames, time_info, status):
        if status:
            print("Status:", status)
        # Append a copy of the incoming block.
        if space_pressed:
            audio_blocks.append(indata.copy())

    # Set up the keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    # Open an input stream using sounddevice.
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
        # Wait until recording is stopped by releasing space
        while is_recording:
            time.sleep(0.05)  # short sleep to reduce CPU usage

    if audio_blocks:
        # Concatenate all recorded blocks into one numpy array.
        recorded_audio = np.concatenate(audio_blocks, axis=0)
        return recorded_audio.flatten()  # flatten to 1D array
    else:
        return None