import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MutiHeadAttention
from layernormal import LayerNoraml
from embedding import TransformerEmbedding

# 前馈神经网络模块
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MutiHeadAttention(d_model=d_model, n_head=n_head)
        self.normalize = LayerNoraml(d_model=d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.dropout2 = nn.Dropout(drop_prob)
    
    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.normalize(x + _x)
        
        _x = x

        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.normalize(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, device, drop_prob=0.1):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, n_head)
            for _ in range(n_layer)
        ])
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# if __name__ == "__main__":
