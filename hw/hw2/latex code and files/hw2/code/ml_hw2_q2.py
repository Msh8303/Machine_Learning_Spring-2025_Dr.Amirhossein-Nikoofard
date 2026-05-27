import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        # Layer 1: 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        # Layer 2: 3x3 conv
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        # Layer 3: 1x1 conv
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

input_tensor = torch.randn(1, 256, 14, 14)

model_64 = BottleneckBlock(256, 64, 256)
params_64 = sum(p.numel() for p in model_64.parameters())
output_64 = model_64(input_tensor)

model_32 = BottleneckBlock(256, 32, 256)
params_32 = sum(p.numel() for p in model_32.parameters())
output_32 = model_32(input_tensor)

print(f"Model with 64 filters:")
print(f"Total Parameters: {params_64}")
print(f"Output Shape: {output_64.shape}")

print(f"\nModel with 32 filters:")
print(f"Total Parameters: {params_32}")
print(f"Output Shape: {output_32.shape}")
