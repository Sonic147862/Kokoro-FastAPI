import os
import uuid
import json
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

# Import your Kokoro backend and settings – adjust the import paths as needed.
from ..inference.kokoro_v1 import KokoroV1
from ..core.config import settings

router = APIRouter()

@router.post("/dev/captioned_speech")
async def captioned_speech(request: Request):
    """
    Endpoint to generate TTS audio and return a header with the path to a JSON file
    containing word timestamp data.
    
    Expected JSON payload structure:
    {
      "input": "Your text here",
      "voice": "voice_path_or_identifier",
      "speed": 1,
      "return_timestamps": true
    }
    """
    payload = await request.json()
    text = payload.get("input")
    voice = payload.get("voice")
    speed = payload.get("speed", 1)
    return_timestamps = payload.get("return_timestamps", False)

    if not text or not voice:
        raise HTTPException(status_code=400, detail="Input and voice fields are required.")

    # Set up a temporary directory to store output files
    base_dir = "temp_files"
    os.makedirs(base_dir, exist_ok=True)

    # Generate a unique identifier for output files
    unique_id = str(uuid.uuid4())

    # Instantiate the Kokoro backend.
    kokoro_backend = KokoroV1()
    if not kokoro_backend.is_loaded:
        # Load the model if not already loaded – assumes settings.model_path is defined.
        try:
            await kokoro_backend.load_model(settings.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model failed to load.")

    # Prepare to collect audio chunks and timestamps
    audio_chunks = []
    combined_timestamps = []

    try:
        # Call the generate method with return_timestamps=True
        async for chunk in kokoro_backend.generate(
            text=text,
            voice=voice,
            speed=speed,
            return_timestamps=return_timestamps
        ):
            # Collect the audio data; in a real scenario you would combine these
            audio_chunks.append(chunk.audio)
            # Add timestamps if present on the chunk
            if return_timestamps and chunk.word_timestamps is not None:
                for ts in chunk.word_timestamps:
                    combined_timestamps.append({
                        "word": ts.word,
                        "start_time": ts.start_time,
                        "end_time": ts.end_time
                    })
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed.")

    # For demonstration, write a dummy MP3 file.
    # In production, combine audio_chunks with your preferred audio library and convert to MP3.
    mp3_file = os.path.join(base_dir, "speech.mp3")
    with open(mp3_file, "wb") as f:
        # Write out a dummy file; replace with valid MP3 encoding logic.
        f.write(b"FAKE_MP3_DATA")

    # Write the combined timestamp data to a JSON file if requested.
    if return_timestamps:
        json_filename = f"{unique_id}.json"
        json_file = os.path.join(base_dir, json_filename)
        with open(json_file, "w") as f:
            json.dump(combined_timestamps, f, indent=2)
    else:
        json_file = ""

    # Create a FileResponse for the MP3 file and attach the x_timestamp_path header
    response = FileResponse(
        mp3_file,
        media_type="audio/mpeg",
        filename="speech.mp3"
    )
    if return_timestamps and json_file:
        # You may return a file system path or a URL if you set up an endpoint for the JSON.
        response.headers["x_timestamp_path"] = json_file
    return response
