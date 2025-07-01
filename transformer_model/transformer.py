import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, 
                src_pad_idx,
                trg_pad_idx,
                enc_voc_size,
                dec_voc_size,
                max_len,
                d_model,
                n_head,
                n_layer,
                ffn_hidden,
                device,
                drop_prob=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, device, drop_prob)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, device, drop_prob)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 目标句子和源句子长度不一致，去掉pad填充的掩码
    # q是目标语言句子，k是源语言句子，pad_idx_q是目标语言填充的索引，pad_idx_k是源语言填充的索引
    # 在没有embedding之前，句子是 [batch_size, time] 的维度
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)  # [batch_size, time]，获取目标句子和源句子的序列长度
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)  # q.ne()是notequal，返回bool矩阵，如果不等于就为True [batch_size, 1, len_q, 1]
        q = q.repeat(1, 1, 1, len_k)                   # [batch_size, 1, len_q, len_k]
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, len_k]
        k = k.repeat(1, 1, len_q, 1)                   # [batch_size, 1, len_q, len_k]
        mask = q&k                                     # [batch_size, 1, len_q, len_k]
        return mask

    # 确保解码器在生成时看到后边的信息，避免未来信息泄露
    def make_causal_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_causal_mask(trg, trg)
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, src_mask)
        return out
