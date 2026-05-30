git add README.mdimport os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Ensure we can import our custom engines regardless of where uvicorn is launched from
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from asr_engine import transcribe_audio
from translation_engine import translate_code_switched_text

app = FastAPI(
    title="Code-Switching NLP Pipeline",
    description="Layer 3 API for ASR + Contextual Translation (Hinglish -> English)",
    version="1.0.0"
)

@app.post("/api/v1/translate")
async def translate_endpoint(audio_file: UploadFile = File(...)):
    start_time = time.time()
    temp_file_path = ""
    
    try:
        # Validate that an actual file was provided
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No audio file uploaded.")
            
        # 1. Save uploaded file to a secure temporary location
        # Using tempfile to avoid race conditions and guarantee uniqueness
        fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd) # Close OS-level handle so we can open it safely in Python
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
            
        # 2. Layer 1: Audio Ingestion and ASR
        # Extract raw phonetic code-switched text
        try:
            raw_text = transcribe_audio(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ASR Engine Error: {str(e)}")
            
        # 3. Layer 2/3: Contextual Translation Routing
        # Transform the raw text into pure English
        try:
            english_translation = translate_code_switched_text(raw_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Translation Engine Error: {str(e)}")
            
        # Calculate total latency
        latency = time.time() - start_time
        
        # 4. Return the unified JSON response
        return JSONResponse(content={
            "raw_text": raw_text,
            "english_translation": english_translation,
            "latency_seconds": round(latency, 2)
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        
    finally:
        # 5. Cleanup: Delete the temporary audio file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to cleanup temporary file {temp_file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
