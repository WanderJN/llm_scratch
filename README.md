# 大模型基本功
## 手撕Transformer模型(transformer_model)
参考链接：https://www.bilibili.com/video/BV1nyyoYLEyL/<br>
![transformer](https://github.com/user-attachments/assets/a3464fe1-3046-44f0-86c6-7ba4bac295a9)

## 手撕GRPO训练器(grpo_trainer_scratch)
### GRPO原理
![grpo_figure](https://github.com/user-attachments/assets/65b9c9d4-c495-40e3-b93e-ab89e2878dae)<br>
对于每个问题 i，GRPO 从旧策略 πθold​​ 中采样一组输出 {i1​,i2​,…,iA​}，然后通过最大化以下目标函数来优化策略模型：<br>
![image](https://github.com/user-attachments/assets/bf53f901-9a9a-44a7-afa2-f412c442fc34)<br>
其中，ϵ 和 β 是超参数，A^i,j​ 是基于组内奖励的相对优势估计。与 PPO 不同，GRPO 通过直接使用奖励模型的输出来估计基线，避免了训练一个复杂的值函数（GRPO的优势是句子粒度的，而非token粒度的）。另外，GRPO 通过直接在损失函数中加入策略模型和参考模型之间的 KL 散度来正则化，而不是在奖励中加入 KL 惩罚项，从而简化了训练过程。<br>
此外，GRPO 通过直接在损失函数中加入策略模型和参考模型之间的 KL 散度来正则化，而不是在奖励中加入 KL 惩罚项，从而简化了训练过程。GRPO使用下面的无偏估计来估计 KL 散度：<br>
![image](https://github.com/user-attachments/assets/7b112f8f-72dc-4a78-b23c-27412fceddb5)<br>
该值一定为正。<br>
<br>
### 流程图像示意
![grpo](https://github.com/user-attachments/assets/6b5c6713-e92f-4c36-bf59-93979f54e19c)
