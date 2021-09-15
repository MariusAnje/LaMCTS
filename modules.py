import torch
from torch import nn
from torch import functional
from torch._C import device
from Functions import SLinearFunction, SConv2dFunction, SBatchNorm2dFunction, SMSEFunction, SCrossEntropyLossFunction    

class STPConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(STPConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return functional.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def create_helper(self):
        self.weightS = nn.Parameter(torch.ones(self.op.weight.size()).requires_grad_())
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.original_w = None
        self.original_b = None
        self.scale = 1.0

    # def set_noise(self, var):
        # self.noise = torch.normal(mean=0., std=var, size=self.noise.size()).to(self.op.weight.device) 
    
    def set_noise(self, var, N, m):
        # N: number of bits per weight, m: number of bits per device
        noise = torch.zeros_like(self.noise).to(self.op.weight.device)
        scale = self.op.weight.abs().max()
        for i in range(1, N//m + 1):
            noise += (torch.normal(mean=0., std=var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
        self.noise = noise.to(self.op.weight.device) * scale
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def mask_indicator(self, method, alpha=None):
        if method == "second":
            return self.weightS.grad.data.abs()
        if method == "magnitude":
            return 1 / (self.op.weight.data.abs() + 1e-8)
        if method == "saliency":
            if alpha is None:
                alpha = 2
            return self.weightS.grad.data.abs() * (self.op.weight.data ** alpha).abs()
        if method == "r_saliency":
            if alpha is None:
                alpha = 2
            # MODIFICATION HERE DUDE
            return self.weightS.grad.abs() * self.weightS.grad.abs().max() / (self.op.weight.data ** alpha + 1e-8).abs() # Original
            # return self.weightS.grad.abs() / (self.op.weight.data ** alpha + 1e-8).abs()
            # return self.weightS.grad.abs() * self.weightS.grad.abs().max() / ((self.op.weight.data * self.scale) ** alpha + 1e-8).abs()
        if method == "subtract":
            return self.weightS.grad.data.abs() - alpha * self.weightS.grad.data.abs() * (self.op.weight.data ** 2)
        else:
            raise NotImplementedError(f"method {method} not supported")

    def set_mask(self, portion, mode):
        if mode == "portion":
            th = self.weightS.grad.abs().view(-1).quantile(1-portion)
            self.mask = (self.weightS.grad.data.abs() <= th).to(torch.float)
        elif mode == "th":
            self.mask = (self.weightS.grad.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def set_mask_mag(self, portion, mode):
        if mode == "portion":
            th = self.op.weight.abs().view(-1).quantile(1-portion)
            self.mask = (self.op.weight.data.abs() <= th).to(torch.float)
        elif mode == "th":
            self.mask = (self.op.weight.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def set_mask_sail(self, portion, mode, method, alpha=None):
        saliency = self.mask_indicator(method, alpha)
        if mode == "portion":
            th = saliency.view(-1).quantile(1-portion)
            self.mask = (saliency <= th).to(torch.float)
        elif mode == "th":
            self.mask = (saliency <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.op.weight)

    def push_S_device(self):
        self.weightS = self.weightS.to(self.op.weight.device)
        self.mask = self.mask.to(self.op.weight.device)

    def clear_S_grad(self):
        with torch.no_grad():
            if self.weightS.grad is not None:
                self.weightS.grad.data *= 0
    
    def fetch_S_grad(self):
        return (self.weightS.grad.abs() * self.mask).sum()
    
    def fetch_S_grad_list(self):
        return (self.weightS.grad.data * self.mask)

    def do_second(self):
        self.op.weight.grad.data = self.op.weight.grad.data / (self.weightS.grad.data + 1e-10)
    
    def normalize(self):
        if self.original_w is None:
            self.original_w = self.op.weight.data
        if (self.original_b is None) and (self.op.bias is not None):
            self.original_b = self.op.bias.data
        scale = self.op.weight.data.abs().max().item()
        self.scale = scale
        self.op.weight.data = self.op.weight.data / scale
        # if self.op.bias is not None:
        #     self.op.bias.data = self.op.bias.data / scale

class SLinear(SModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = SLinearFunction.apply

    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x * self.scale, xS * self.scale, (self.op.weight + self.noise) * self.mask, self.weightS)
        # x = self.scale * x
        # xS = self.scale * xS
        if self.op.bias is not None:
            x += self.op.bias
        if self.op.bias is not None:
            xS += self.op.bias
        return x, xS

class SConv2d(SModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply

    def forward(self, xC):
        x, xS = xC
        # x, xS = self.function(x, xS, (self.op.weight + self.noise) * self.mask, self.weightS, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        x, xS = self.function(x * self.scale, xS * self.scale, (self.op.weight + self.noise) * self.mask, self.weightS, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        if self.op.bias is not None:
            x += self.op.bias.reshape(1,-1,1,1).expand_as(x)
        if self.op.bias is not None:
            xS += self.op.bias.reshape(1,-1,1,1).expand_as(xS)
        return x, xS

class SSTPConv2d(SModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply

    def forward(self, xC):
        x, xS = xC
        weight = self.op.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        x, xS = self.function(x * self.scale, xS * self.scale, (weight + self.noise) * self.mask, self.weightS, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        if self.op.bias is not None:
            x += self.op.bias
        if self.op.bias is not None:
            xS += self.op.bias
        return x, xS

class NModule(nn.Module):
    # def set_noise(self, var):
    #     self.noise = torch.normal(mean=0., std=var, size=self.noise.size()).to(self.op.weight.device) 
    def set_noise(self, var, N, m):
        noise = torch.zeros_like(self.noise)
        scale = self.op.weight.abs().max()
        for i in range(1, N//m + 1):
            noise += torch.normal(mean=0., std=var, size=self.noise.size()) * (pow(2, - i*m))
        self.noise = noise.to(self.op.weight.device) * scale
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)

class NLinear(NModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.noise = torch.zeros_like(self.op.weight)
        self.function = nn.functional.linear

    def forward(self, x):
        x = self.function(x, (self.op.weight + self.noise), self.op.bias)
        return x

class NConv2d(NModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise = torch.zeros_like(self.op.weight)
        self.function = nn.functional.conv2d

    def forward(self, x):
        x = self.function(x, (self.op.weight + self.noise), self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x

class NSTPConv2d(NModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise = torch.zeros_like(self.op.weight)
        self.function = nn.functional.conv2d

    def forward(self, x):
        weight = self.op.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        x = self.function(x, (weight + self.noise), self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x

class SReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.op = nn.ReLU(inplace)
    
    def forward(self, xC):
        x, xS = xC
        with torch.no_grad():
            mask = (x > 0).to(torch.float)
        return self.op(x), xS * mask

class SMaxpool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        return_indices = True
        self.op = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def parse_indice(self, indice):
        bs, ch, w, h = indice.shape
        length = w * h
        BD = torch.LongTensor(list(range(bs))).expand([length,ch,bs]).swapaxes(0,2).reshape(-1)
        CD = torch.LongTensor(list(range(ch))).expand([bs,length,ch]).swapaxes(1,2).reshape(-1)
        shape = [bs, ch, -1]
        return shape, [BD, CD, indice.view(-1)]

    
    def forward(self, xC):
        x, xS = xC
        x, indices = self.op(x)
        shape, indices = self.parse_indice(indices)
        xS = xS.view(shape)[indices].view(x.shape)
        return x, xS

class SBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.op = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.function = SBatchNorm2dFunction.apply
    
    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x, xS, self.op.running_mean, self.op.running_var, self.op.weight, self.op.bias, self.op.training, self.op.momentum, self.op.eps)
        return x, xS

class FakeSModule(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        if isinstance(self.op, nn.MaxPool2d):
            self.op.return_indices = False
    
    def forward(self, xC):
        x, xS = xC
        x = self.op(x)
        return x, None

class SModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.push_S_device()

    def clear_S_grad(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.clear_S_grad()

    def do_second(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.do_second()

    def fetch_S_grad(self):
        S_grad_sum = 0
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                S_grad_sum += m.fetch_S_grad()
        return S_grad_sum
    
    def calc_S_grad_th(self, quantile):
        S_grad_list = None
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                if S_grad_list is None:
                    S_grad_list = m.fetch_S_grad_list().view(-1)
                else:
                    S_grad_list = torch.cat([S_grad_list, m.fetch_S_grad_list().view(-1)])
        th = torch.quantile(S_grad_list, 1-quantile)
        # print(th)
        return th
    
    def calc_sail_th(self, quantile, method, alpha=None):
        sail_list = None
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                sail = m.mask_indicator(method, alpha).view(-1)
                if sail_list is None:
                    sail_list = sail
                else:
                    sail_list = torch.cat([sail_list, sail])
        th = torch.quantile(sail_list, 1-quantile)
        # print(th)
        return th

    def set_noise(self, var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
                # m.set_noise(var)
                mo.set_noise(var, N, m)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.clear_noise()
    
    def set_mask(self, th, mode):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.set_mask(th, mode)
    
    def set_mask_mag(self, th, mode):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.set_mask_mag(th, mode)
    
    def set_mask_sail(self, th, mode, method, alpha=None):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.set_mask_sail(th, mode, method, alpha)
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.clear_mask()
    
    def to_fake(self, device):
        for name, m in self.named_modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, SMaxpool2D) or isinstance(m, SReLU):
                new = FakeSModule(m.op)
                self._modules[name] = new
        self.to(device)
    
    def normalize(self):
        for mo in self.modules():
            if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
                if mo.original_w is None:
                    mo.original_w = mo.op.weight.data
                if (mo.original_b is None) and (mo.op.bias is not None):
                    mo.original_b = mo.op.bias.data
                scale = mo.op.weight.data.abs().max().item()
                mo.op.weight.data = mo.op.weight.data / scale
                if mo.op.bias is not None:
                    mo.op.bias.data = mo.op.bias.data / scale
    
    def de_normalize(self):
        for mo in self.modules():
            if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
                if mo.original_w is None:
                    raise Exception("no original weight")
                else:
                    mo.op.weight.data = mo.original_w
                    if mo.original_b is not None:
                        mo.op.bias.data = mo.original_b
    
    def back_real(self, device):
        for name, m in self.named_modules():
            if isinstance(m, FakeSModule):
                if isinstance(m.op, nn.Linear):
                    if m.op.bias is not None:
                        bias = True
                    new = SLinear(m.op.in_features, m.op.out_features, bias)
                    new.op = m.op
                    self._modules[name] = new

                elif isinstance(m.op, nn.Conv2d):
                    if m.op.bias is not None:
                        bias = True
                    new = SConv2d(m.op.in_channels, m.op.out_channels, m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.groups, bias, m.op.padding_mode)
                    new.op = m.op
                    self._modules[name] = new

                elif isinstance(m.op, nn.MaxPool2d):
                    new = SMaxpool2D(m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.return_indices, m.op.ceil_mode)
                    new.op = m.op
                    new.op.return_indices = True
                    self._modules[name] = new

                elif isinstance(m.op, nn.ReLU):
                    new = SReLU()
                    new.op = m.op
                    self._modules[name] = new

                else:
                    raise NotImplementedError
        self.to(device)