import mirdata
import librosa
import numpy as np
import os

print("Verifying GuitarSet...")
guitarset = mirdata.initialize('guitarset')
guitarset.download()  # Just checks the index file now

os.makedirs("processed_data", exist_ok=True)
STRING_TUNINGS = [40, 45, 50, 55, 59, 64]

tracks = list(guitarset.load_tracks().items())
total_tracks = len(tracks)

print(f"Starting extraction for all {total_tracks} tracks. Grab a coffee! ☕")

# REMOVED the [:5] slice! It now processes everything.
for i, (track_id, track) in enumerate(tracks):
    print(f"Processing {i+1}/{total_tracks}: {track_id}")
    
    # 1. Convert Audio to CQT Heatmap (using 44100 Hz to avoid Nyquist error)
    y, sr = librosa.load(track.audio_mic_path, sr=44100)
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('E2'), n_bins=192, bins_per_octave=24))
    
    # 2. Blank Labels (20 = String is silent)
    labels = np.full((6, cqt.shape[1]), 20)
    
    # 3. Match Frets to Time Frames
    if hasattr(track, 'notes') and track.notes:
        string_names = list(track.notes.keys()) 
        for string_idx, string_name in enumerate(string_names):
            string_notes = track.notes[string_name]
            if string_notes is None or len(string_notes.intervals) == 0: continue
            
            for interval, midi_pitch in zip(string_notes.intervals, string_notes.notes):
                start_frame = int(librosa.time_to_frames(interval[0], sr=sr, hop_length=512))
                end_frame = int(librosa.time_to_frames(interval[1], sr=sr, hop_length=512))
                fret_played = int(round(midi_pitch)) - STRING_TUNINGS[string_idx]
                
                if 0 <= fret_played < 20:
                    labels[string_idx, start_frame:end_frame] = fret_played
                    
    np.savez_compressed(f"processed_data/{track_id}.npz", cqt=cqt, labels=labels)

print("Full dataset preprocessing complete!")