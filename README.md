- [大模型基本功](#大模型基本功)
  - [手撕Transformer模型(transformer\_model)](#手撕transformer模型transformer_model)
  - [手撕DPO训练器(dpo\_trainer\_scratch)](#手撕dpo训练器dpo_trainer_scratch)
    - [DPO原理](#dpo原理)
  - [手撕PPO训练器(ppo\_trainer\_scratch)](#手撕ppo训练器ppo_trainer_scratch)
    - [RLHF-PPO完整流程](#rlhf-ppo完整流程)
    - [PPO原理](#ppo原理)
  - [手撕GRPO训练器(grpo\_trainer\_scratch)](#手撕grpo训练器grpo_trainer_scratch)
    - [GRPO原理](#grpo原理)
    - [GRPO流程图像示意](#grpo流程图像示意)
    - [GRPO训练时Loss从0开始并且上升](#grpo训练时loss从0开始并且上升)
      - [解释1：TRL库的Loss计算原因](#解释1trl库的loss计算原因)
      - [解释2：GRPO默认策略是单步更新](#解释2grpo默认策略是单步更新)
  - [手撕DAPO训练器(dapo\_trainer\_scratch)](#手撕dapo训练器dapo_trainer_scratch)
    - [DAPO相较于GRPO的优化](#dapo相较于grpo的优化)

# 大模型基本功
## 手撕Transformer模型(transformer_model)
参考链接：https://www.bilibili.com/video/BV1nyyoYLEyL/<br>
<img src="https://github.com/user-attachments/assets/a3464fe1-3046-44f0-86c6-7ba4bac295a9" width="40%"/><br>

## 手撕DPO训练器(dpo_trainer_scratch)
### DPO原理
学习参考：https://zhuanlan.zhihu.com/p/1888312479307772555<br>
![image](https://github.com/user-attachments/assets/344c9cdf-1160-43ac-be3c-a111e49bbf8f)<br>
DPO是一种直接优化人类偏好的方法，主要用于语言模型对齐：直接从偏好数据（chosen、rejected数据对）中学习，无需显式的奖励模型。通过对比学习的方式，让模型更倾向于生成人类偏好的回答
工作流程：<br>
1. 收集偏好数据对 (preferred, rejected)<br>
2. 使用Bradley-Terry模型建模偏好概率<br>
3. 直接优化策略模型，使其输出偏好回答的概率更高<br>
![image](https://github.com/user-attachments/assets/68a1ac6f-a661-43ba-ab4b-4b615205cbbe)<br>
`loss = -log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))`<br>
相较于PPO，DPO在训练过程中是一种单阶段直接优化的方式，其本质和SFT监督微调一致，只是将Loss计算修改为了与chosen、rejected输出概率累计的。其训练流程简单，稳定性好，但模型训练上限不如PPO。<br>

## 手撕PPO训练器(ppo_trainer_scratch)
### RLHF-PPO完整流程
<img src="https://github.com/user-attachments/assets/c3f21b6c-9a89-4cb4-8589-fc677afe2633" width="70%"/><br>
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
### GRPO流程图像示意
![grpo](https://github.com/user-attachments/assets/6b5c6713-e92f-4c36-bf59-93979f54e19c)
### GRPO训练时Loss从0开始并且上升
参考链接：https://github.com/huggingface/open-r1/issues/239<br>
#### 解释1：TRL库的Loss计算原因
在TRL库中，默认是新旧策略的概率值per_token_logps是一样的，`torch.exp(per_token_logps - per_token_logps.detach())`一直为1，但是仍然存在梯度。参考解释如下图：<br>
![image](https://github.com/user-attachments/assets/bdae05be-0768-4efc-bc3e-a62f952123ce)<br>
**为什么可以这样做**：因为在deepseek-math的论文中提到：“The policy model only has a single update following each exploration stage."，他只采样了一次。所以，就直接这么写了。<br>
#### 解释2：GRPO默认策略是单步更新
现在GRPO实现中策略都是单步更新，导致新旧策略是一样的，所以重要性采样系数是1，然后优势函数A是一个组当中每个reward的标准化，那么对优势函数A求期望自然也就是0了。所以GRPO的loss实际上就是新旧策略的KL散度项再乘一个系数beta，这也就是为什么训练过程中loss曲线和KL散度曲线分布如此相似，因为只差了一个系数beta。具体推导原因如下图所示：<br>
![image](https://github.com/user-attachments/assets/94d4d1e1-c2e0-474c-8e53-3c4058f88ee2)
![image](https://github.com/user-attachments/assets/4cd494fe-8c3c-4e78-94c2-b6d868c67de9)

## 手撕DAPO训练器(dapo_trainer_scratch)
### DAPO相较于GRPO的优化
1. **移除KL散度**<br>
   在RLHF场景中，强化学习的目标是对其模型的输出，使得输出更符合人类偏好，这时候不应该偏离原始模型太远，需要KL散度来限制。但是在强化学习训练长思维链推理模型时，需要模型更加自由的去探索，模型分布可能与初始模型显著偏离，KL散度的限制是没有必要的。
2. **Clip-Higher（上限更高）**
   在PPO和GRPO中对于新旧策略概率比进行裁剪，限定范围为固定范围（1-epsilon, 1+epsilon, epsilon一般取0.2），防止模型更新幅度过大。<br>
   举例：对于两个动作（token），假设概率分别为0.9（高概率）和0.01（低概率），那么在更新之后，两个token的最大概率分别为0.9\*1.2=1.08，0.01\*0.12=0.012。这意味着对于高概率的token，受到的约束反而更小，低概率token受到的约束更大，想要实现概率显著增加非常困难，限制了模型探索的多样性。<br>
   DAPO对上裁剪和下裁剪的范围解耦，增加上裁剪的范围，给低概率token更多的探索空间。<br>
3. **动态采样**<br>
   GRPO中，通过组内奖励的均值和标准差计算优势，但是如果一个样本的组内奖励全部为0或1，这时候优势为0，零优势导致策略更新时没有梯度，无法优化模型，样本效率不高。<br>
   DAPO的做法是在训练前不断进行采样，直到一个batch内的所有样本的组内奖励既不全为0也不全为1。<br>
4. **Token级策略梯度损失**<br>
   GRPO的损失函数中，每个样本的损失是对样本内的token损失求平均，这样忽略长度的影响，即短样本和长样本对最终损失的贡献是相同的权重。而长样本具有更多的token，这样导致长样本中的token对整体损失的贡献很低，可能无法对高质量的长样本进行奖励，以及对低质量短样本进行惩罚。<br>
   DAPO损失函数的做法：对组内所有token求平均（**从样本级提升到token级**），确保长序列中的每个token都能对梯度更新产生同等影响（长样本因为拥有更多的token，其整体会对最终的损失贡献更大）。这一改进不仅提升了训练稳定性，还避免了过长响应中的低质量模式。<br>
   ![image](https://github.com/user-attachments/assets/38151427-05a3-4096-8799-2195dd1f7b77)<br>
5. **过长的奖励塑形**<br>
   在RL训练中，过长的响应通常会被截断，并受到惩罚。然而，这种惩罚可能会引入奖励噪声，干扰训练过程。DAPO提出了软过长惩罚机制，通过长度感知的惩罚区间，逐步增加对过长响应的惩罚，从而减少奖励噪声并稳定训练。<br>
   当响应长度超过预定义长度时最大值时，我们定义一个惩罚区间。在此间隔内，响应时间越长，则它受到惩罚。这种惩罚被添加到原始的基于规则的正确性奖励中，从而发出信号以避免过长的响应。<br>
   ![image](https://github.com/user-attachments/assets/11dacf1e-2c37-4420-8d01-a0e845aac5b2)<br>

   
