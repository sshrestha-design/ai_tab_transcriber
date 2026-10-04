import torch
import librosa
import numpy as np
import torch.nn as nn
import scipy.signal
import sys

# 1. Same neural network architecture
class TabCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(1, 32, (3, 3), padding=1), nn.Conv2d(32, 64, (3, 3), padding=1)
        self.pool, self.relu, self.flatten = nn.MaxPool2d((2, 2)), nn.ReLU(), nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(64 * 48 * 2, 128), self.relu, nn.Dropout(0.5))
        self.heads = nn.ModuleList([nn.Linear(128, 21) for _ in range(6)])
        
    def forward(self, x):
        x = self.fc(self.flatten(self.pool(self.relu(self.conv2(self.pool(self.relu(self.conv1(x))))))))
        return [head(x) for head in self.heads]

# 2. Load the trained brain
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TabCNN().to(device)
model.load_state_dict(torch.load("tabcnn_model.pth", weights_only=True))
model.eval()

# 3. Load YOUR audio file
# Change this string to the name of your MP3 or WAV file
AUDIO_FILE = "my_riff.mp3" 

print(f"Loading {AUDIO_FILE}...")
try:
    y, sr = librosa.load(AUDIO_FILE, sr=44100)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{AUDIO_FILE}'. Make sure it is in the same folder as this script!")
    sys.exit(1)

# 4. Convert to Spectrogram & Transcribe
print("Converting audio to visual spectrogram...")
cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('E2'), n_bins=192, bins_per_octave=24))

print("AI is transcribing...")
cqt_padded = np.pad(cqt, ((0,0), (4, 4)))
predictions = np.zeros((6, cqt.shape[1]), dtype=int)

for i in range(cqt.shape[1]):
    window = cqt_padded[:, i:i+9]
    x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(x_tensor)
        for string_idx in range(6):
            predictions[string_idx, i] = torch.argmax(outputs[string_idx], dim=1).item()

# 5. Smooth the glitches
for string_idx in range(6):
    predictions[string_idx] = scipy.signal.medfilt(predictions[string_idx], kernel_size=11)

# 6. Print the Tab
print("\n" + "="*60)
print(f"🎸 AI GENERATED TAB FOR: {AUDIO_FILE} 🎸")
print("="*60)

string_names = ['E', 'A', 'D', 'G', 'B', 'e']

# Print the first ~10 seconds of the song (up to 800 frames)
for s in range(5, -1, -1):
    line = f"{string_names[s]}|-"
    for frame in range(0, min(800, predictions.shape[1]), 10):
        fret = predictions[s, frame]
        line += f"{fret if fret < 20 else '-'}-"
    print(line)