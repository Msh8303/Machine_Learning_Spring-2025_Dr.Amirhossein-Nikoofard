import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
input_tensor = torch.randn(1, 1, 5, 5, requires_grad=True) # (Batch, Channel, H, W)
target = torch.randn(1, 1, 3, 3)
kernel_size = 3

W = nn.Parameter(torch.randn(kernel_size, kernel_size))
M = nn.Parameter(torch.randn(kernel_size, kernel_size))

W_final = W * M 

patches = F.unfold(input_tensor, kernel_size=(kernel_size, kernel_size))
flat_weight = W_final.view(1, -1)
output_flat = torch.matmul(flat_weight, patches)
output = output_flat.view(1, 1, 3, 3)

criterion = nn.MSELoss()
loss = criterion(output, target)
loss.backward()

print(f"Loss: {loss.item():.4f}")
print("Gradients for W:")
print(W.grad)
print("\nGradients for M:")
print(M.grad)