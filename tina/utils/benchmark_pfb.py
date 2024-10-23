import torch
import numpy as np
import time
from tina.layers.fft_layer import FFTLayer

def generate_win_coeffs(M, P, window_fn="hamming"):
    from scipy.signal import get_window, firwin
    win_coeffs = get_window(window_fn, M*P)
    sinc = firwin(M * P, cutoff=1.0 / P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs

def benchmark_pfb_layer(input_size, M, P):
    """
    Benchmark a Polyphase Filter Bank (PFB) using FFT optimized by TINA.
    """
    # Generate random input data
    x = np.random.rand(input_size).astype(np.float32)
    
    # Generate window coefficients
    win_coeffs = generate_win_coeffs(M, P)
    
    # Convert input to PyTorch tensor
    x_tensor = torch.from_numpy(x).unsqueeze(0)
    
    # Initialize TINA FFT layer
    fft_layer = FFTLayer(input_size)
    
    # Time the FFT operation
    start_time = time.time()
    output = fft_layer(x_tensor)
    elapsed_time = time.time() - start_time
    
    print(f"FFT layer executed in {elapsed_time:.6f} seconds.")
    return elapsed_time

if __name__ == "__main__":
    # Example usage
    input_size = 1024
    M = 16
    P = 256
    benchmark_pfb_layer(input_size, M, P)
