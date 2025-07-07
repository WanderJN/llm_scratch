# 大模型基本功
## 手撕Transformer模型(transformer_model)
参考链接：https://www.bilibili.com/video/BV1nyyoYLEyL/<br>
![transformer](https://github.com/user-attachments/assets/a3464fe1-3046-44f0-86c6-7ba4bac295a9)<br>

## 手撕PPO训练器(ppo_trainer_scratch)
### RLHF-PPO完整流程
![RLHF-PPO](https://github.com/user-attachments/assets/c3f21b6c-9a89-4cb4-8589-fc677afe2633)<br>
![PPO](https://github.com/user-attachments/assets/16a95022-9811-4570-ac1a-390ba703814e)<br>
### PPO原理
PPO 的目标是通过最大化以下替代目标函数来优化策略模型：
![image](https://github.com/user-attachments/assets/0717554c-00f5-4a78-b1eb-21f838377628)<br>
其中，πθ​ 和 πθ_old​​ 分别是当前策略模型和旧策略模型，q 和 o 是从问题数据集和旧策略 πθ_old​​ 中采样的问题和输出。超参数 ϵ 用于稳定训练过程。优势 A_i​ 是通过广义优势估计（GAE）计算的，计算过程基于奖励 {ri≥j​} 和学习到的值函数 Vπold​​。为了减轻对奖励模型的过度优化，标准方法是在每个标记的奖励中添加一个来自参考模型的每个标记的KL惩罚，即：<br>
![image](https://github.com/user-attachments/assets/b4f6e67d-7c30-4d00-a9ee-30baf2d6cd7e)<br>

其中，r是奖励模型，π_ref是参考模型，通常是初始的监督微调（SFT）模型，而 β 是 KL 惩罚项的系数。<br>
给定价值函数V和奖励函数R，At使用广义优势估计(GAE)计算：<br>
![image](https://github.com/user-attachments/assets/79a72031-278b-4d0f-a65b-2efbd7b65e62)<br>
![image](https://github.com/user-attachments/assets/b4203cc6-2b3d-495a-a13f-c114aab42778)<br>
**PPO存在的问题**：PPO 中的值函数通常是一个与策略模型大小相当的模型，这带来了显著的内存和计算负担。此外，在 LLMs 的上下文中，值函数在训练过程中被用作优势计算中的Baseline，但通常只有最后一个 token 会被奖励模型赋予奖励分数，这可能使得值函数的训练变得复杂。<br>

## 手撕GRPO训练器(grpo_trainer_scratch)
### GRPO原理
![grpo_figure](https://github.com/user-attachments/assets/65b9c9d4-c495-40e3-b93e-ab89e2878dae)<br>
与PPO相比，GRPO消除了价值函数，并以组相对的方式估计优势。对于特定的问题-答案对(q, a)，行为策略 πθ_old​​ 会抽样一组G个单独的回答o_i{i=1到G}。然后，第i个回答的优势通过归一化组维度的奖励Ri{i=1到G}来计算：<br>
![image](https://github.com/user-attachments/assets/7c6dcc52-9c1c-4f85-a8e4-b6f97b80a9b6)<br>
对于每个问题 i，GRPO 从旧策略 πθold​​ 中采样一组输出 {i1​,i2​,…,iA​}，然后通过最大化以下目标函数来优化策略模型：<br>
![image](https://github.com/user-attachments/assets/bf53f901-9a9a-44a7-afa2-f412c442fc34)<br>
其中，ϵ 和 β 是超参数，A^i,j​ 是基于组内奖励的相对优势估计。与 PPO 不同，GRPO 通过直接使用奖励模型的输出来估计基线，避免了训练一个复杂的值函数（GRPO的优势是句子粒度的，而非token粒度的）。<br>
另外，GRPO 通过直接在损失函数中加入策略模型和参考模型之间的 KL 散度来正则化，而不是在奖励中加入 KL 惩罚项，从而简化了训练过程。GRPO使用下面的无偏估计来估计 KL 散度：<br>
![image](https://github.com/user-attachments/assets/7b112f8f-72dc-4a78-b23c-27412fceddb5)<br>
该值一定为正。<br>
<br>
### 流程图像示意
![grpo](https://github.com/user-attachments/assets/6b5c6713-e92f-4c36-bf59-93979f54e19c)
