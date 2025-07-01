import torch
import torch.nn as nn
from transformer import Transformer

# 定义词汇表
src_vocab = ['<pad>', 'I', 'love', 'you', 'too']
trg_vocab = ['<pad>', 'Je', 't\'aime', 'aussi']

# 创建词汇表映射
src_word2idx = {word: idx for idx, word in enumerate(src_vocab)}
trg_word2idx = {word: idx for idx, word in enumerate(trg_vocab)}

# 示例句子
src_sentences = ["I love you", "I love you too"]
trg_sentences = ["Je t'aime", "Je t'aime aussi"]

# 最大长度
max_len = 5

# 将句子转换为索引
def sentences_to_indices(sentences, word2idx, max_len):
    indices = []
    for sentence in sentences:
        tokens = sentence.split()
        indices.append([word2idx.get(token, 0) for token in tokens] + [0] * (max_len - len(tokens)))
    return torch.tensor(indices, dtype=torch.long)

# 准备输入数据
src_data = sentences_to_indices(src_sentences, src_word2idx, max_len)
trg_data = sentences_to_indices(trg_sentences, trg_word2idx, max_len)

# 定义模型参数
src_pad_idx = src_word2idx['<pad>']
trg_pad_idx = trg_word2idx['<pad>']
enc_voc_size = len(src_vocab)
dec_voc_size = len(trg_vocab)
d_model = 512
n_head = 8
n_layer = 6
ffn_hidden = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = Transformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    enc_voc_size=enc_voc_size,
    dec_voc_size=dec_voc_size,
    max_len=max_len,
    d_model=d_model,
    n_head=n_head,
    n_layer=n_layer,
    ffn_hidden=ffn_hidden,
    device=device
).to(device)

# 将数据移动到设备
src_data = src_data.to(device)
trg_data = trg_data.to(device)

# 前向传播
output = model(src_data, trg_data)

print(output)