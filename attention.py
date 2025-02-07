import torch
import torch.nn as nn
import math

class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MutiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)  # 将多头结果融合
        self.softmax = nn.Softmax(dim=-1)
    
    # 这里传入的qkv其实都是x，计算后可以得到qkv，其目的是为了计算交叉注意力时方便
    def forward(self, q, k, v, mask=None):
        batch_size, time, d_model = q.shape
        head_dim = self.d_model//self.n_head      # n_d表示每个头的维度大小，将d_model的维度切分给每个头
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) 
        q = q.view(batch_size, time, self.n_head, head_dim).permute(0,2,1,3)  # [batch_size, self.n_head, time, head_dim]
        k = k.view(batch_size, time, self.n_head, head_dim).permute(0,2,1,3)
        v = v.view(batch_size, time, self.n_head, head_dim).permute(0,2,1,3)
        # 计算注意力分数
        score = q@k.transpose(2, 3) / math.sqrt(head_dim)   # [batch_size, self.n_head, time, time] 这里是每个时刻输入与其他时刻输入计算得到的注意力分数，所以是[time, time]矩阵
        # 判断是否使用掩码
        if mask is not None:
            # 在掩码为0的位置的得分填充为一个极小的值，以便于后续softmax后约为0
            score = score.masked_fill(mask==0, -1e-9)
        
        score = self.softmax(score)@v                   # [batch_size, self.n_head, time, head_dim]
        score = score.permute(0,2,1,3).contiguous().view(batch_size, time, d_model) # [batch_size, time, d_model]
        output = self.w_combine(score)      # [batch_size, time, d_model]
        return output

if __name__ == "__main__":
    d_model = 512
    n_head = 4
    attention = MutiHeadAttention(d_model, n_head)
    input = torch.rand(128, 32, d_model)    # [batch_size, time, d_model]
    out = attention(input)
    print(out.shape)