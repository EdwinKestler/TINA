# tina/__init__.py

"""
TINA - A Python library for optimized layers and utilities.
Designed for efficient integration with PyTorch and ONNX models.

Modules:
    layers: Optimized layers (FFT, Convolution).
    utils: Utility functions for data handling and preprocessing.
"""

# Import key components for easy access
from .layers.fft_layer import FFTLayer
from .layers.conv_layer import ConvLayer
from .utils.data_loader import load_data

__all__ = ["FFTLayer", "ConvLayer", "load_data"]
