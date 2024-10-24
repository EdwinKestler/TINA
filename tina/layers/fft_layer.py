import torch
import torch.nn as nn
import numpy as np

class FFTLayer(nn.Module):
    """
    FFTLayer - A custom PyTorch layer that applies Fast Fourier Transform (FFT)
    to the input tensor. The layer supports both forward and inverse FFT operations.

    Arguments:
    input_size (int): The size of the input data (number of features).
    inverse (bool): If True, performs inverse FFT (iFFT) instead of FFT. Default is False.
    """

    def __init__(self, input_size, inverse=False):
        super(FFTLayer, self).__init__()
        self.input_size = input_size
        self.inverse = inverse

    def forward(self, x):
        """
        Applies FFT or inverse FFT to the input tensor.

        Arguments:
        x (torch.Tensor): Input tensor with shape (batch_size, input_size).

        Returns:
        torch.Tensor: The result of the FFT operation, with the same shape as the input.
        """
        if not torch.is_tensor(x):
            raise ValueError("Input must be a PyTorch tensor.")
        if x.size(1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(1)}.")

        if self.inverse:
            # Perform inverse FFT (iFFT)
            output = torch.fft.ifft(x, dim=1)
        else:
            # Perform forward FFT
            output = torch.fft.fft(x, dim=1)

        return output

    def extra_repr(self):
        return f"input_size={self.input_size}, inverse={self.inverse}"
