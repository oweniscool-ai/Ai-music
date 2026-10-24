import os
import sys
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from audiocraft.models import MusicGen

# Read the prompt sent from your iPhone
song_prompt = sys.argv[1] if len(sys.argv) > 1 else "chill lofi study beats"

print(f"Starting free cloud generation for prompt: {song_prompt}")

# Leverage open-source weights optimized for free infrastructure tiers
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.to('cpu') # GitHub Actions utilizes powerful multi-core CPUs

chunk_duration = 30
total_duration = 180 # 3 minutes total
sample_rate = model.sample_rate

# 1. Generate base audio movement
model.set_generation_params(duration=chunk_duration)
current_wav = model.generate([song_prompt])
compiled_audio = [current_wav.squeeze()]

loops_needed = (total_duration // chunk_duration) - 1

# 2. Free Audio Continuation Loop Engine
for i in range(loops_needed):
    print(f"Processing chunk {i+2}/6...")
    overlap_samples = int(5 * sample_rate)
    audio_prompt_segment = current_wav[..., -overlap_samples:]
    
    current_wav = model.generate_continuation(
        prompt_tokens=audio_prompt_segment,
        prompt_sample_rate=sample_rate,
        descriptions=[song_prompt],
        progress=False
    )
    
    new_content = current_wav.squeeze()[overlap_samples:]
    compiled_audio.append(new_content)

# 3. Stitch chunks together cleanly
print("Stitching song matrix...")
final_wav = torch.cat(compiled_audio, dim=-1).numpy()

# Normalize sound limits to protect speaker hardware
if np.max(np.abs(final_wav)) > 0:
    final_wav = final_wav / np.max(np.abs(final_wav))

# Save the final file directly into your website's output folder
os.makedirs("docs", exist_ok=True)
output_path = "docs/my_generated_song.wav"
wavfile.write(output_path, sample_rate, (final_wav * 32767).astype(np.int16))
print(f"Success! Song saved natively to: {output_path}")
