import torch
import unittest
from titans_pytorch.neuro.baselines.mamba2 import Mamba2
from titans_pytorch.neuro.baselines.gla import GatedLinearAttention
from titans_pytorch.neuro.baselines.griffin import Griffin

class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.D = 1, 30, 128
        self.input_tensor = torch.randn(self.B, self.T, self.D)

    def test_mamba2_shape(self):
        print("\nTesting Mamba-2 Forward Pass...")
        model = Mamba2(d_model=self.D, d_state=16, d_conv=4, expand=2)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.B, self.T, self.D))
        print("Mamba-2 Shape OK")

    def test_gla_shape(self):
        print("\nTesting Gated Linear Attention Forward Pass...")
        model = GatedLinearAttention(d_model=self.D, n_head=4)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.B, self.T, self.D))
        print("GLA Shape OK")

    def test_griffin_shape(self):
        print("\nTesting Griffin (Hybrid) Forward Pass...")
        # Griffin combines GLA/Recurrent block with local attention
        model = Griffin(dim=self.D, depth=2)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.B, self.T, self.D))
        print("Griffin Shape OK")

if __name__ == '__main__':
    unittest.main()
