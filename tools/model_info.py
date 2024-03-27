import sys
sys.path.append('./')
import torch
from thop import profile, clever_format
from models.ReadCurrent import ReadCurrent

input = torch.randn(1, 3000)
model = ReadCurrent([32, 64, 128, 256, 512], n_fc_neurons=2048, depth=29, shortcut=True)

flops, params = profile(model, inputs=(input, ))
print(f'flops: {flops},  params: {params}')
flops, params = clever_format([flops, params], "%.3f")
print(f'format flops: {flops},  format params: {params}')