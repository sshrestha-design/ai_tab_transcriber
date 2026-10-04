import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

# 1. Use Apple Silicon GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# 2. PyTorch Dataset Loader
class GuitarDataset(Dataset):
    def __init__(self, data_dir, context_frames=9):
        self.context = context_frames
        half = context_frames // 2
        self.X, self.Y = [], []
        
        for file in glob.glob(f"{data_dir}/*.npz"):
            data = np.load(file)
            cqt, labels = data['cqt'], data['labels']
            
            # Pad time dimension to safely grab 9-frame windows
            cqt_padded = np.pad(cqt, ((0, 0), (half, half)))
            
            for i in range(labels.shape[1]):
                self.X.append(cqt_padded[:, i : i + self.context])
                self.Y.append(labels[:, i])
                
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.int64)
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        # Add channel dimension (1 channel, like a grayscale image)
        return np.expand_dims(self.X[idx], axis=0), self.Y[idx]

# 3. Define the Neural Network Architecture
class TabCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Math: 192 bins / 4 = 48. 9 frames / 4 = 2.
        self.fc = nn.Sequential(
            nn.Linear(64 * 48 * 2, 128),
            self.relu,
            nn.Dropout(0.5)
        )
        
        # 6 Output Heads (One for each string E, A, D, G, B, e)
        self.heads = nn.ModuleList([nn.Linear(128, 21) for _ in range(6)])
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.fc(self.flatten(x))
        return [head(x) for head in self.heads]

# 4. Training Execution
print("Loading dataset into RAM...")
dataset = GuitarDataset("processed_data")
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print(f"Loaded {len(dataset)} audio frames.")

model = TabCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 8# Keep it low for a quick test run
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        
        # Combine the error (loss) from all 6 strings
        loss = sum(criterion(outputs[i], batch_y[:, i]) for i in range(6))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

# 5. Save the brain!
torch.save(model.state_dict(), "tabcnn_model.pth")
print("Model saved to tabcnn_model.pth!")