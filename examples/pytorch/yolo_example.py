import torch
import torch.nn as nn
from utils.hardware_utils import detect_hardware

# Define a simple YOLO-style network (as an example)
class YOLOModel(nn.Module):
    def __init__(self):
        super(YOLOModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)  # Example final layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# Load a YOLO-style model
model = YOLOModel()

# Detect available hardware and switch execution mode
hardware = detect_hardware()

if hardware == "nvidia":
    print("Running on NVIDIA GPU with CUDA")
    model = model.to("cuda")

elif hardware == "amd":
    print("Running on AMD ROCm")
    model = model.to("cuda")  # Assuming ROCm uses the same API

elif hardware == "mps":
    print("Running on Apple Silicon (M1/M2)")
    model = model.to("mps")

else:
    print("Running on CPU")
    model = model.to("cpu")

# Example input
input_data = torch.randn(1, 3, 32, 32)

# Run inference
output = model(input_data.to(model.device))

# Print the result
print(f"Model output: {output}")
