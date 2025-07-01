import torch
import torch.nn as nn

class LayerNoraml(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNoraml, self).__init__()
        self.gama = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # 在d_model维度计算均值和方差，因为不同batch长度不同，同一位置的元素也毫无关系，因此将各个元素自己对自己归一化
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        out = (x - mean)/torch.sqrt(var + self.eps)
        out = self.gama * out + self.beta
        return out
