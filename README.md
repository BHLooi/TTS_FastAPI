# TTS_FastAPI
TTS using FastAPI and Kokoro

This repo contains two folder:  
- **Code**  
- **deployment**

## Code
Folder contents:
* Kokoro-82M-v1.0-ONNX - TTS model in onnx type.  
    * Download the model from huggingface (https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main)
* mapping.json - Mapping tokens for TTS.
* tts.py - FastAPI script
* Dockerfile - For dockerization.
* requirements.txt - Required packages.


Dockerize steps:
```bash
cd "<your-path>/Code"
docker build -t tts:kokoro .
```

## deployment
Folder contents:
* config - Folder containing the config.py  
    * Different models can be use according to what is available.
    * Source: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/onnx
* docker-compose.yaml - To run the container.

Deployment Steps:
```bash
cd "<your-path>/deployment"
docker-compose -f docker-compose.yaml up
```

## Container API
There are three APIs:
* /health - If API is running.
```bash
// Example
curl -X GET "http://localhost:5011/health"
```

* /model_info - Displays the model information, including the voice options available.
```bash
// Example
curl -X GET "http://localhost:5011/model_info"
```

* /generate_audio - Main API
```bash
// Example
curl -X POST "http://localhost:5011/generate_audio" \
     -H "Content-Type: application/json" \
     -d '{"text": "How are you? How are you today?", "voice": "emma"}' \
     --output generate_audio.wav
```
