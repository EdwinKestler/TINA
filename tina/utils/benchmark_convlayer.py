import time
import torch
from tina.layers.conv_layer import ConvLayer

def benchmark_convlayer(device, in_channels=3, out_channels=16, input_size=(1, 3, 224, 224)):
    # Initialize ConvLayer with TINA optimizations
    conv_layer = ConvLayer(in_channels, out_channels, kernel_size=3, use_tina_optim=True).to(device)
    
    # Create a random input tensor
    input_tensor = torch.randn(input_size).to(device)

    # If using NVIDIA optimization (Tensor Cores), convert input tensor to half-precision (float16)
    if device == "cuda" and conv_layer.use_tina_optim:
        input_tensor = input_tensor.half()

    # Run the forward pass multiple times and time it
    num_runs = 100
    torch.cuda.synchronize() if device == "cuda" else None  # Synchronize GPU before timing
    start_time = time.time()
    for _ in range(num_runs):
        output = conv_layer(input_tensor)
    torch.cuda.synchronize() if device == "cuda" else None  # Synchronize GPU after timing
    
    avg_time = (time.time() - start_time) / num_runs
    print(f"Average forward pass time on {device}: {avg_time:.6f} seconds")

# Run the benchmark on CPU and GPU if available
if __name__ == "__main__":
    print("Benchmarking ConvLayer with TINA optimizations...")
    benchmark_convlayer(device="cpu")
    if torch.cuda.is_available():
        benchmark_convlayer(device="cuda")
