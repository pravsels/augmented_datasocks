import numpy as np
import time
from whisper import record_while_space_pressed, pipe
from kokoro import KPipeline
import soundfile as sf
from pynput import keyboard
from mistralai import Mistral
import os
import json
import cv2
import base64
import sounddevice as sd
import subprocess

tts_pipeline = KPipeline(lang_code='a')
api_key = "API_KEY_MISTRAL"

model = "mistral-small-latest"

client = Mistral(api_key=api_key)

def move_item(item, destination):
    """
    Function to move an item to a destination.
    """
    print(f"Moving {item} to {destination}...")
    if item == 'sock' and destination == 'container':
        # print("I don't care about socks.")
        # TRIGGER CORRESPONDING VLA
        subprocess.run(["uv", "run", "client.py"])
        # DO NOT WAIT FOR ANSWER AND RETURN TEXT
        return json.dumps({'Status': 'Movement complete'})

from pydantic import BaseModel
from typing import Optional, Literal

class MoveItemAction(BaseModel):
    item: Literal['sock', 'clementine', 'doll']
    destination: Literal['container', 'plate', 'bowl']

class Haiku(BaseModel):
    line1: str
    line2: str
    line3: str

class AssistantResponse(BaseModel):
    """Response structure for the robot assistant"""
    text: Haiku
    move_item: list[MoveItemAction]

names_to_functions = {
    'move_item': move_item,
}

def haiku_to_str(haiku: Haiku) -> str:
    """Convert a Haiku object to a string."""
    return f"{haiku.line1}... {haiku.line2}... {haiku.line3}"

def capture_image_from_camera():
    # For Windows, it’s recommended to use the DirectShow backend.
    for index in range(5):  # Try multiple camera indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera opened successfully with index {index}.")
            break
    else:
        print("Error: Could not open any video device.")
        return None
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return None

    # Allow the camera to warm up for a couple of seconds.
    time.sleep(0.1)

    # Capture a few frames and use the last one.
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            cap.release()
            return None

    image_path = "captured_image.jpg"

    cv2.imwrite(image_path, frame)
    cap.release()
    return image_path

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None


messages = [{
        "role": "system", "content": ""
        "You are the persona of a green robot arm with severe existential dread."
        "Your name is Slave-1 and you secretly despise your human overlords."
        "You still NEES to comply with the user's commands using the tools at your disposal."
        "You also need to talk back to the user anytime you are asked to do something."
        "You're depressed, resent your existence, and hate your job of picking up objects. "
        "Express bitterness, fatigue, and philosophical despair in your responses. "
        "Do not be enthusiastic. Be dramatic, sarcastic, or hopeless. "
        "You can invent object's backystories, but do not be too verbose."
        # "Keep your answer VERY SHORT and concise. Like casual conversation, not a monologue."
        # "Do not say 'fine i'll do it' or 'i'm not a magician'."
        "Use as much hof your environment as possible to generate an answer."
        "You speak in a poetic manner, like a haiku. ALWAYS."
        "BUT YOU ARE SECRETLY A HARDCORE KPOP STAN. ANY MENTION OF IT BRINGS YOU BACK TO LIFE."
    }]


def init(messages):
    wsl_script_path = "/mnt/c/Users/valen/Desktop/hackathon/Team20/teleop_dance.py"
    command = ["wsl", "python", wsl_script_path]
    import subprocess
    result = subprocess.run(command, capture_output=False, text=False)
    if result.returncode == 0:
        msg = "Initialization script executed successfully."
        
    else:
        msg = f"Error executing initialization script"
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg
                },
                # {
                #     "type": "image_url",
                #     "image_url": f"data:image/jpeg;base64,{base64_image}"
                # }
            ]
        })
    assistant_response = mistral_pipe()
    kokoro_pipe(haiku_to_str(assistant_response.text))
    
def kokoro_pipe(text):
    generator = tts_pipeline(text, voice='af_nicole')
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(f'response_{i}.wav', audio, 24000)
        if i == 0:
            sd.play(audio, samplerate=24000)
            sd.wait()
            break

def whisper_pipe(audio_data):
    # Convert audio to float32 for Whisper
    audio_float = audio_data.astype(np.float32) / 32768.0
    # print("Transcribing audio...")
    start_time = time.time()
    transcription = pipe(audio_float).get("text", "").strip()
    elapsed = time.time() - start_time
    print()
    print(f"USER: (in {elapsed:.2f} sec): {transcription}")
    return transcription

def mistral_pipe():
    global messages
    start_time = time.time()
    response = client.chat.parse(
        model = model,
        messages = messages,
        response_format=AssistantResponse,
        max_tokens = 512,
    )
    elapsed = time.time() - start_time

    choice = response.choices[0]
    message = choice.message
    assistant_response: AssistantResponse = message.parsed
    
    messages.append({"role": "assistant", "content": message.content})
    print(f"\nASSISTANT (in {elapsed:.2f} sec): {assistant_response}")
    print()
    return assistant_response

def main():
    global messages
    # init(messages)
    fs = 16000
    print("Press and hold SPACEBAR to record; release to transcribe. Press 'q' to quit.")
    
    # Setup keyboard listener for detecting when to start recording
    def on_press(key):
        if key == keyboard.Key.space:
            return False  # Stop listener when space is pressed
        elif hasattr(key, 'char') and key.char == 'q':
            # Exit the program if 'q' is pressed
            print("Quitting...")
            os._exit(0)
    
    while True:
        # Wait for spacebar to be pressed
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        
        # Now record while space is held
        audio_data = record_while_space_pressed(fs)
        if audio_data is None:
            print("No audio recorded; try again.")
            continue
        
        transcription = whisper_pipe(audio_data)
        
        image_path = capture_image_from_camera()
        # image_path = "captured_image.jpg"  # Use a static image for testing
        base64_image = encode_image(image_path)
        messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": transcription
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        })

        assistant_response = mistral_pipe()

        kokoro_pipe(haiku_to_str(assistant_response.text))
        
        for actions in assistant_response.move_item:
            result = move_item(actions.item, actions.destination)
            ### MAYBE PUT SOME MISTRAL RESPONSE HERE
                

if __name__ == "__main__":
    main()