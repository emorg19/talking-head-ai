# extract_audio.py - Extract audio features from a video
import os
import torchaudio
import ffmpeg

def extract_audio(video_path, output_audio_path):
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(output_audio_path, format="wav", acodec="pcm_s16le", ar="44100")
        .run(overwrite_output=True)
    )
    print(f"Extracted audio saved to {output_audio_path}")