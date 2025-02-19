from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from config.config import config
from pydantic import BaseModel, Field
import os
import numpy as np
from onnxruntime import InferenceSession
import json
from misaki import en, espeak
import scipy.io.wavfile as wavfile
import uvicorn
import requests
import random

# deletion of file
def delete_file(file_path: str):
    """Delete the file after the response has been sent."""
    if os.path.exists(file_path):
        os.remove(file_path)

# Load misaki model
fallback = espeak.EspeakFallback(british=False) # en-us
g2p = en.G2P(trf=False, british=False, fallback=fallback) # no transformer, American English

# Load mapping json file
with open("mapping.json","r", encoding="utf-8") as file:
    mapping_data = json.load(file)

mapping_vocab = mapping_data['vocab']
def phonemes_to_ids(phonemes, mapping):
    return [mapping[p] for p in phonemes if p in mapping]

# Function to create tokens
def create_tokens(text):
    phonemes, tokens = g2p(text)
    tokens = phonemes_to_ids(phonemes, mapping_vocab)
    tokens = [[0, 0, *tokens, 0, 0]]
    return tokens

# Load and initialize onnx model
model_path = "./Kokoro-82M-v1.0-ONNX/onnx/model.onnx" # Options: model.onnx, model_fp16.onnx, model_quantized.onnx, model_q8f16.onnx, model_uint8.onnx, model_uint8f16.onnx, model_q4.onnx, model_q4f16.onnx
sess = InferenceSession(model_path)

# Load voice options
folder_path = './Kokoro-82M-v1.0-ONNX/voices'
files = os.listdir(folder_path)  # Get list of files and directories
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]  # Filter out directories
voice_options = [('_'.join(i.split('_')[1:]) if len(i.split('_'))>1 else i).replace('.bin','') for i in files]

class TextRequest(BaseModel):
    text: str
    voice: str

app = FastAPI(title="Kokoro TTS API")

@app.post("/generate_audio")
async def generate_audio(request: TextRequest, background_tasks: BackgroundTasks):
    try:
        voice_option = request.voice
        voices = np.fromfile(f'./Kokoro-82M-v1.0-ONNX/voices/af_{voice_option}.bin', dtype=np.float32).reshape(-1, 1, 256)
        print(f"Using {voice_option} voice.", flush=True)
    except:
        voices = np.fromfile(f'./Kokoro-82M-v1.0-ONNX/voices/af.bin', dtype=np.float32).reshape(-1, 1, 256)
        print("No voice option, defaulting to af voice.", flush=True)

    text = request.text
    print(text, flush=True)
    tokens = create_tokens(text)
    ref_s = voices[len(tokens)]

    try:
        audio = sess.run(None, dict(
            input_ids=tokens,
            style=ref_s,
            speed=np.array([0.9], dtype=np.float32),
        ))[0]
        
        # Save audio to temporary file
        rand_id = str(int(random.random()*(10^12)))
        output_audio_path = f"audio_{rand_id}.wav"
        wavfile.write(output_audio_path, 24000, audio[0])

        background_tasks.add_task(delete_file, output_audio_path)
            
        # Return audio file
        return FileResponse(
            output_audio_path,
            media_type="audio/wav",
            filename=output_audio_path
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating audio: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": sess is not None}

@app.get("/model_info")
async def get_model_info():
    return {
        "model_name": "Kokoro-82M",
        "voices": voice_options
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5011)