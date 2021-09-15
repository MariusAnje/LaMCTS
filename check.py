import torch
import Soperations
import modules
from torch import nn

def check_same(x: torch.Tensor, y: torch.Tensor):
    return (x - y).abs().max().item() / max(x.abs().max().item(), y.abs().max().item())


criteria = Soperations.SCrossEntropyLoss()
nnC = nn.CrossEntropyLoss()
# model = Soperations.SepConv(3,3,3,1,1)
model = modules.SConv2d(3,3,3,1,1,
                      groups=3, bias=False)
nnModel = nn.Conv2d(3,3,3,1,1,groups=3, bias=False)
nnModel.weight.data = model.op.weight.data
GT = torch.LongTensor([1,0])
x = torch.randn(2,3,3,3)
xS = torch.ones_like(x)
output, outputS = model((x,xS))
nnOutput = nnModel(x)
output = output.sum(dim=1).sum(dim=1)
outputS = outputS.sum(dim=1).sum(dim=1)
nnOutput = nnOutput.sum(dim=1).sum(dim=1)
loss = criteria(output, outputS, GT)
loss.backward()
loss1 = nnC(nnOutput, GT)
loss1.backward()
print(check_same(output, nnOutput))
print(check_same(model.op.weight.grad, nnModel.weight.grad))