import torch.nn as nn
from attention import MutiHeadAttention
from layernormal import LayerNoraml
from embedding import TransformerEmbedding
from encoder import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MutiHeadAttention(d_model, n_head)
        self.normalize1 = LayerNoraml(d_model=d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.cross_attention = MutiHeadAttention(d_model, n_head)
        self.normalize2 = LayerNoraml(d_model=d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.normalize3 = LayerNoraml(d_model=d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    # dec和enc分别是解码和编码的输入，t_mask是target mask表示解码时不能看到未来的信息,
    # s_mask是source mask表示计算交叉注意力时，忽略编码中的某些部分
    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        x = self.normalize1(x + _x)

        _x = x

        # encoder和decoder交叉的注意力机制, q来自于解码器，k和v来自于编码器
        x = self.cross_attention(x, enc, enc, s_mask)
        x = self.dropout2(x)
        x = self.normalize2(x + _x)

        _x = x

        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.normalize3(x + _x)

        return x


class Decoder(nn.Module):
    def __init__(self,dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer ,device, drop_prob=0.1):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len,drop_prob, device)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ffn_hidden, n_head,drop_prob)
            for _ in range(n_layer)
        ])
        self.last_fc = nn.Linear(d_model, dec_voc_size)
    
    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            x = layer(dec, enc, t_mask, s_mask)
        x = self.last_fc(x)

        return x


