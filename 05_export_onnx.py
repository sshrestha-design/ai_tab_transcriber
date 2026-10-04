import torch
import torch.nn as nn

# 1. Redefine the exact architecture
class TabCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(1, 32, (3, 3), padding=1), nn.Conv2d(32, 64, (3, 3), padding=1)
        self.pool, self.relu, self.flatten = nn.MaxPool2d((2, 2)), nn.ReLU(), nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(64 * 48 * 2, 128), self.relu, nn.Dropout(0.5))
        self.heads = nn.ModuleList([nn.Linear(128, 21) for _ in range(6)])
        
    def forward(self, x):
        x = self.fc(self.flatten(self.pool(self.relu(self.conv2(self.pool(self.relu(self.conv1(x))))))))
        # ONNX prefers tuples over lists for multiple output heads
        return tuple(head(x) for head in self.heads) 

print("Loading PyTorch model...")
model = TabCNN()
model.load_state_dict(torch.load("tabcnn_model.pth", map_location="cpu", weights_only=True))
model.eval()

# 2. Create a dummy input to show ONNX the shape of our data [Batch, Channel, Height, Width]
dummy_input = torch.randn(1, 1, 192, 9)

print("Compiling to ONNX format...")
torch.onnx.export(
    model, 
    dummy_input, 
    "tabcnn.onnx",               # The optimized output file
    input_names=['input'], 
    output_names=['E', 'A', 'D', 'G', 'B', 'e'],
    # This tells ONNX that the batch size can change, letting us process a whole song at once!
    dynamic_axes={'input': {0: 'batch_size'}, 
                  'E': {0: 'batch_size'}, 'A': {0: 'batch_size'}, 'D': {0: 'batch_size'}, 
                  'G': {0: 'batch_size'}, 'B': {0: 'batch_size'}, 'e': {0: 'batch_size'}}
)

print("✅ Successfully exported tabcnn.onnx!")