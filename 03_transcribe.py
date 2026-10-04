import torch
import librosa
import numpy as np
import mirdata
import torch.nn as nn
import scipy.signal  # <-- NEW: For the median filter

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TabCNN().to(device)
model.load_state_dict(torch.load("tabcnn_model.pth", weights_only=True))
model.eval()

print("Loading test audio...")
guitarset = mirdata.initialize('guitarset')
track = list(guitarset.load_tracks().values())[0] 

y, sr = librosa.load(track.audio_mic_path, sr=44100)
cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('E2'), n_bins=192, bins_per_octave=24))

print("Transcribing...")
cqt_padded = np.pad(cqt, ((0,0), (4, 4)))
predictions = np.zeros((6, cqt.shape[1]), dtype=int)

for i in range(cqt.shape[1]):
    window = cqt_padded[:, i:i+9]
    x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(x_tensor)
        for string_idx in range(6):
            predictions[string_idx, i] = torch.argmax(outputs[string_idx], dim=1).item()

# <-- NEW: Apply Median Filter Smoothing -->
# A kernel size of 11 smooths over ~250 milliseconds, wiping out tiny glitches
for string_idx in range(6):
    predictions[string_idx] = scipy.signal.medfilt(predictions[string_idx], kernel_size=11)

print("\n" + "="*60)
print("🎸 AI GENERATED ASCII TAB (Smoothed) 🎸")
print("="*60)

string_names = ['E', 'A', 'D', 'G', 'B', 'e']

for s in range(5, -1, -1):
    line = f"{string_names[s]}|-"
    for frame in range(0, min(400, predictions.shape[1]), 10):
        fret = predictions[s, frame]
        line += f"{fret if fret < 20 else '-'}-"
    print(line)