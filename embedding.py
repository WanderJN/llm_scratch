import torch
import torch.nn as nn
import torch.nn.functional as F

# 将词汇表索引转化为输入模型的矩阵
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)  # 索引为1的词汇视为填充值，这样可以初始化为0


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionEmbedding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)  # 因为pos要进行运算返回给self.encoding，所以需要作为数组 [max_len, 1]

        _2i = torch.arange(0, d_model, step=2, device=device).float()   # [d_model/2]

        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))  # [max_len, d_model/2]
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))  # [max_len, d_model/2]
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]           # [seq_len, d_model]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(x)
        return self.drop_out(tok_emb + pos_emb)

if __name__ == "__main__":
    embedding = TransformerEmbedding(10, 4, 512, 0.2, "cpu")
    input = torch.LongTensor([[1,2,3,2], [4,5,3,1]])
    e = embedding(input)
    print(e.shape)
    print(e)