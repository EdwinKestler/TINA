import torch
import torch.nn as nn
import numpy as np

class FFTLayer(nn.Module):
    def __init__(self, input_size):
        super(FFTLayer, self).__init__()
        self.input_size = input_size
    
    def forward(self, x):
        # Perform FFT operation on the input tensor
        fft_result = torch.fft.fft2(x, dim=(-2, -1))
        return fft_result
