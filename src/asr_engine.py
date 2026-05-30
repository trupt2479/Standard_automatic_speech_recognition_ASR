import os
import wave
import math
import struct
from faster_whisper import WhisperModel

def generate_dummy_audio(filepath="dummy.wav"):
    """Generates a 1-second 440Hz sine wave dummy audio file."""
    sample_rate = 16000
    num_samples = sample_rate * 1  # 1 second
    amplitude = 32767
    
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            value = int(amplitude * math.sin(2 * math.pi * 440.0 * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

def transcribe_audio(audio_path: str):
    print("Loading faster-whisper medium model...")
    # Load the medium model. compute_type="int8" and device="cpu" is used for compatibility, 
    # but could be upgraded to float16/cuda if GPU is available.
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    
    print(f"Transcribing {audio_path}...")
    # task="transcribe" and vad_filter=True are required to capture raw phonetic strings
    segments, info = model.transcribe(audio_path, task="transcribe", vad_filter=True)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability}")
    
    transcription = []
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        transcription.append(segment.text)
        
    return " ".join(transcription)

if __name__ == "__main__":
    dummy_file = "dummy_test.wav"
    print("Generating dummy audio file...")
    generate_dummy_audio(dummy_file)
    
    print("Running transcription test...")
    try:
        text = transcribe_audio(dummy_file)
        print("\nFinal Transcription:")
        print(text)
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
            print("Cleaned up dummy audio file.")
