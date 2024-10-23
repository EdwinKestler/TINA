import unittest
import torch
import torch.nn as nn
from tina.layers.conv_layer import ConvLayer

class TestConvLayer(unittest.TestCase):

    def setUp(self):
        # Setup a ConvLayer for testing
        self.in_channels = 1
        self.out_channels = 32
        self.kernel_size = 3
        self.conv_layer = ConvLayer(in_channels=self.in_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.kernel_size)
        self.test_input = torch.rand(1, self.in_channels, 28, 28)  # Example input tensor

    def test_conv_output_shape(self):
        """
        Test if the ConvLayer produces the correct output shape.
        """
        output = self.conv_layer(self.test_input)
        expected_output_shape = (1, self.out_channels, 28, 28)  # Same padding assumed in the layer
        self.assertEqual(output.shape, expected_output_shape, "Output shape mismatch")

    def test_conv_weight_initialization(self):
        """
        Test if the ConvLayer weights are initialized correctly.
        """
        # Check that weights have been initialized and are not None
        self.assertIsNotNone(self.conv_layer.conv.weight, "Weights not initialized")

    def test_conv_gradients(self):
        """
        Test if ConvLayer supports gradient backpropagation.
        """
        input_tensor = self.test_input.clone().requires_grad_(True)
        output = self.conv_layer(input_tensor)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(input_tensor.grad, "Gradients were not calculated")
    
    def test_conv_layer_with_custom_padding(self):
        """
        Test ConvLayer with custom padding.
        """
        conv_layer_with_padding = ConvLayer(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            padding=1)  # Custom padding
        output = conv_layer_with_padding(self.test_input)
        expected_output_shape = (1, self.out_channels, 28, 28)  # Custom padding keeps dimensions
        self.assertEqual(output.shape, expected_output_shape, "Output shape mismatch with padding")

    def test_conv_layer_on_non_square_input(self):
        """
        Test ConvLayer on non-square input images.
        """
        non_square_input = torch.rand(1, self.in_channels, 32, 64)  # Non-square input tensor
        output = self.conv_layer(non_square_input)
        expected_output_shape = (1, self.out_channels, 32, 64)  # Expect same input dimensions with default padding
        self.assertEqual(output.shape, expected_output_shape, "Output shape mismatch for non-square input")

if __name__ == "__main__":
    unittest.main()
