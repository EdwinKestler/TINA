import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.hardware_utils import detect_hardware



from tina.layers.conv_layer import ConvLayer
from tina.layers.fft_layer import FFTLayer
from tina.utils.benchmark_pfb import benchmark_pfb_layer

# Define the simple neural network architecture using TINA layers
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Using the TINA-optimized ConvLayer
        self.conv1 = ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fft_layer = FFTLayer(input_size=64 * 7 * 7)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fft_layer(x)  # Apply FFT layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Load and preprocess the MNIST dataset
def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Training loop
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')

# Testing loop
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main function to execute training and testing
def main():
    batch_size = 64
    epochs = 10
    learning_rate = 0.01
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize model and optimizer
    model = MNISTModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Benchmark the FFT Layer performance
    print("Benchmarking FFT Layer:")
    benchmark_pfb_layer(input_size=64 * 7 * 7, M=16, P=256)

    # Train and test the model
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == "__main__":
    main()
