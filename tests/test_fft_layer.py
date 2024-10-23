import unittest
import torch
import numpy as np
from tina.layers.fft_layer import FFTLayer

class TestFFTLayer(unittest.TestCase):
    
    def setUp(self):
        # Setup an example input tensor for tests
        self.input_size = 1024
        self.fft_layer = FFTLayer(input_size=self.input_size)
        self.test_input = torch.rand(1, self.input_size)
    
    def test_fft_output_shape(self):
        """
        Test if FFTLayer produces the correct output shape.
        """
        output = self.fft_layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape, "Output shape mismatch")
    
    def test_fft_is_complex(self):
        """
        Test if the FFTLayer output is complex.
        """
        output = self.fft_layer(self.test_input)
        self.assertTrue(torch.is_complex(output), "Output is not complex")

    def test_fft_values(self):
        """
        Test if FFTLayer produces correct values compared to NumPy.
        """
        # Convert input tensor to numpy
        input_np = self.test_input.squeeze(0).numpy()
        expected_output = np.fft.fft(input_np)

        # Get FFTLayer output and convert to numpy
        output = self.fft_layer(self.test_input).detach().numpy().squeeze(0)
        
        # Assert if the output matches expected FFT values from NumPy
        np.testing.assert_almost_equal(output, expected_output, decimal=5, err_msg="FFT values mismatch")
    
    def test_fft_layer_on_empty_input(self):
        """
        Test if FFTLayer raises an error on an empty input tensor.
        """
        empty_input = torch.tensor([])
        with self.assertRaises(ValueError):
            self.fft_layer(empty_input)

    def test_fft_layer_on_non_contiguous_input(self):
        """
        Test if FFTLayer handles non-contiguous input.
        """
        non_contiguous_input = self.test_input[:, ::2]
        output = self.fft_layer(non_contiguous_input)
        self.assertEqual(output.shape, non_contiguous_input.shape, "Output shape mismatch for non-contiguous input")
    
    def test_fft_gradients(self):
        """
        Test if FFTLayer supports gradient backpropagation.
        """
        input_tensor = self.test_input.clone().requires_grad_(True)
        output = self.fft_layer(input_tensor)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(input_tensor.grad, "Gradients were not calculated")

if __name__ == "__main__":
    unittest.main()
