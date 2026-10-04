import os
# Bypass the macOS matplotlib system font cache bug
os.environ["MPL_IGNORE_SYSTEM_FONTS"] = "1"

import gradio as gr
import librosa
import numpy as np
import onnxruntime as ort
import scipy.signal
import subprocess

print("Loading ONNX engine...")
session = ort.InferenceSession("tabcnn.onnx")

def generate_tab(audio_filepath):
    if audio_filepath is None: 
        return "Please upload an audio file."
    
    # 1. Silently convert any uploaded file (like .m4a) to .wav using ffmpeg
    clean_wav_path = audio_filepath + ".wav"
    subprocess.run(["ffmpeg", "-y", "-i", audio_filepath, clean_wav_path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Load Audio (now guaranteed to be a readable WAV file)
    y, sr = librosa.load(clean_wav_path, sr=44100)
    
    # 3. Convert to Spectrogram
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('E2'), n_bins=192, bins_per_octave=24))
    
    # 4. Prepare ALL frames at once for maximum speed
    cqt_padded = np.pad(cqt, ((0,0), (4, 4)))
    num_frames = cqt.shape[1]
    
    # Extract sliding windows into a single massive batch array
    windows = np.array([cqt_padded[:, i:i+9] for i in range(num_frames)])
    x_input = np.expand_dims(windows, axis=1).astype(np.float32) # Shape: [num_frames, 1, 192, 9]
    
    # 5. Transcribe the entire song in ONE millisecond step
    outputs = session.run(None, {'input': x_input})
    
    # 6. Process outputs & smooth glitches
    predictions = np.zeros((6, num_frames), dtype=int)
    for string_idx in range(6):
        predictions[string_idx] = np.argmax(outputs[string_idx], axis=1)
        predictions[string_idx] = scipy.signal.medfilt(predictions[string_idx], kernel_size=11)

    # 7. Format neatly wrapped ASCII Tab
    string_names = ['E', 'A', 'D', 'G', 'B', 'e']
    tab_output = "🎸 AI GENERATED TAB\n" + "="*80 + "\n\n"
    
    frames_per_row = 150 # How many time-steps to show before wrapping to a new line
    
    for chunk_start in range(0, min(1500, num_frames), frames_per_row):
        chunk_end = min(chunk_start + frames_per_row, num_frames)
        for s in range(5, -1, -1):
            line = f"{string_names[s]} |-"
            for frame in range(chunk_start, chunk_end, 5): # Sample every 5 frames
                fret = predictions[s, frame]
                line += f"{str(fret).ljust(2, '-') if fret < 20 else '--'}-"
            tab_output += line + "\n"
        tab_output += "\n" # Blank line between blocks
        
    return tab_output

# 8. Define the Web UI
interface = gr.Interface(
    fn=generate_tab,
    # NEW: Added sources parameter to enable the mic
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or Record (Clean Solo Guitar)"),
    outputs=gr.Code(language="markdown", label="Generated ASCII Tab"), 
    title="🎸 Local AI Guitar Tab Transcriber",
    description="Powered by ONNX Runtime. Upload an audio file or record directly, and the AI will transcribe it instantly.",
    theme="monochrome"
)

if __name__ == "__main__":
    interface.launch()

