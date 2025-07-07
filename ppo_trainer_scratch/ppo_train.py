from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import random
import torch
import torch.nn as nn

class PromptDataset(Dataset):
    def __init__(self, prompts: List[str], tokenizer: AutoTokenizer, apply_chat_template: bool = False):
        self.prompts = prompts
        self.tokenizer = tokenizer

        self.final_prompts = []

        for prompt in self.prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize = False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
            
            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.final_prompts[idx]


# 价值模型，用于预测每一步生成token的动作产生的预估长期收益。使用actor模型初始化，外加一个回归头，输入shape为(batch_size, seq_len, 1)
class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()  # Freeze the base model
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, num_actions):
        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        value_model_output = self.value_head(hidden_state)
        # 只提取出response部分的价值
        values = value_model_output.squeeze(-1)[:, -num_actions:]
        return values


# 创建经验池，存储了之前采样的经验（价值、优势等）。每次采样后，将新的经验添加到池中，直到达到限制。可以从池中随机抽取批次进行训练。
class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    
    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = {
            "seqs", "action_log_probs", "values", "returns", "advantages", "attention_mask", "action_masks", "num_actions"
        }
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
        
        # 如果buffer已满，移除最旧的经验
        if len(self.buffer) + len(batch) > self.limit:
            self.buffer = self.buffer[len(batch):]
        self.buffer.extend(batch)      # 添加新经验到buffer中

    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


# 存储一次采样的结果
@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, None]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor

# 存储一次采样计算得到的经验
@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_masks: torch.Tensor
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None

# 存储一个batch下生成优势和回报等信息
@dataclass
class BufferItem:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]

# 将生成的一批经验重新组装好格式，用于训练
def collate_fn(batch):

    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    
    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)
    
    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))


@dataclass
class PPOArguments:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = "./output"
    log_dir = "./runs"
    # 一共迭代多少轮
    epoch = 3
    # 生成一次经验，训练的轮数
    num_iterations = 5
    # 一次从提示词数据集中取多少条数据用于生成经验
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2
    # PPO的epsilon值，控制策略更新的范围
    clip_eps = 0.2
    # 折扣因子和 GAE参数
    gamma = 0.1
    lambd = 0.2

    # 学习率
    actor_lr = 1e-5
    critic_lr = 1e-5

    # KL控制系数，这个系数越大，KL散度对奖励的影响越大
    kl_ctl = 0.1
    # 奖励值的裁剪范围
    clip_reward_value = 0.2


class PPOTrainer:
    def __init__(self, actor_model: AutoModelForCausalLM,
                 actor_tokenizer: AutoTokenizer,
                 ref_model: AutoModelForCausalLM,
                 reward_model: AutoModelForCausalLM,
                 reward_tokenizer: AutoTokenizer,
                 prompts_dataset: Dataset,
                 args: PPOArguments = None):
        self.args = args

        self.actor_model = actor_model
        self.actor_tokenizer = actor_tokenizer
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.critic_model = Critic(self.actor_model.base_model).to(device=self.args.device)

        # 填充方式为左填充
        self.actor_tokenizer.padding_side = 'left'
        self.eos_token_id = actor_tokenizer.eos_token_id
        self.pad_token_id = actor_tokenizer.pad_token_id

        # 加载其他配置信息
        self.optimizer_actor = torch.optim.Adam(self.actor_model.parameters(), lr=self.args.actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic_model.parameters(), lr=self.args.critic_lr)
        self.writer = SummaryWriter(self.args.log_dir)
        self.prompts_dataloader = DataLoader(prompts_dataset, batch_size=self.args.rollout_batch_size, shuffle=True)


    # 对prompts列表进行采样
    def generate_samples(self, promots, model, max_length, max_new_tokens, n_samples_per_prompt, micor_rollout_batch_size):
        # n_samples_per_prompt 表示每个prompt生成的样本数量
        # micor_rollout_batch_size 表示微步采样的批次大小（将几个放在一个batchsize去生成采样结果）

        samples_list = []
        model.eval()
        all_prompts = sum([[prompt]*n_samples_per_prompt for prompt in promots], [])
        for i in range(0, len(all_prompts), micor_rollout_batch_size):
            batch_prompts = all_prompts[i:i + micor_rollout_batch_size]
            inputs = self.actor_tokenizer(batch_prompts, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
            input_ids = inputs.input_ids

            seqs = model.generate(
                **inputs.to(self.args.device),
                max_new_tokens=max_new_tokens,
                eos_token_id = self.eos_token_id, 
                pad_token_id = self.pad_token_id
            )
            # 超过最大长度就截断，不足最大长度，就padding
            if seqs.size(1) >= max_new_tokens + max_length:
                seqs = seqs[:, :max_new_tokens + max_length]
            else:
                seqs = torch.cat([seqs, torch.full((seqs.size(0),max_new_tokens + max_length - seqs.size(1)), fill_value=self.pad_token_id, device=seqs.device)], dim=1)

            attention_mask = (seqs.ne(self.pad_token_id)).to(dtype=torch.long)
            ans = seqs[:, input_ids.size(1):]
            action_mask = (ans.ne(self.eos_token_id) & ans.ne(self.pad_token_id)).to(dtype=torch.long)

            samples = Samples(
                seqs=seqs,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )
            samples_list.append(samples)

        return samples_list


    # 计算动作的log概率
    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        # 计算策略模型输出token的概率
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # 获取除最后一个元素外的全部概率值
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)                 # shape: [batch_size, seq_len-1, vocab_size]
        # 提取出正确答案位置的概率 ([:, 1:]表示向右移动了一位，正好对应答案位置)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))   # shape: [batch_size, seq_len-1, 1]
        # 只提取response部分的概率值
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]    # shape: [batch_size, num_actions]
        return action_log_probs

    # 计算kl散度
    def compute_approx_kl(self, action_log_probs, ref_action_log_probs, action_mask=None):
        """
        :return: 近似KL散度，shape: [batch_size, num_actions]
        """
        log_ratio = action_log_probs.float() - ref_action_log_probs.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask

        return log_ratio
    
    # 计算实际奖励，（reward包括了reward和kl散度：reward = clip_reward + kl）
    def compute_rewards(self, kl, rewards_score, action_mask, kl_ctl=0.1, clip_reward_value=0.2):
        kl_divergence_estimate  = -kl_ctl * kl
        rewards = kl_divergence_estimate               # 初始reward为kl散度，[batch_size, num_actions]

        if not isinstance(clip_reward_value, torch.Tensor):
            clip_reward_value = torch.tensor(clip_reward_value, device=self.args.device)
        
        # 奖励值裁剪
        rewards_clip = torch.clamp(rewards_score, -clip_reward_value, clip_reward_value)
        
        ends = action_mask.sum(1) + 1   # 找到最后有效的位置的下标（因为有padding），[batch_size]
        # 把batchsize里每个样本的最后有效位置上叠加reward（clip之后的reward），其余位置均为kl散度值
        batch_size = rewards.size()[0]
        for j in range(batch_size):
            rewards[j, :ends[j]][-1] += rewards_clip[j, 0]

        return rewards
        

    # 计算优势和回报
    # A(t) = R(t) + gamma * V(t+1) - V(t)
    # GAE: A(t) = A(t) + (gamma * lambd) * A(t+1)  =  R(t) + gamma * V(t+1) - V(t) + gamma * lambd * A(t+1)
    # 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, G(T+1) = 0，则 A(T) = R(T) - V(T)，得出A(T)
    # A(T-1) = [R(T-1) + gamma * V(T) - V(T-1)] + gamma * lambd * A(T)，知道A(T)即可得出A(T-1)
    # 以此类推，直到A(0) = [R(0) + gamma * V(1) - V(0)] + gamma * lambd * A(1)，得出A(0)
    # 回报G(t) = A(t) + V(t)
    def get_advantages_and_returns(self, values, rewards, action_mask, gamma, lambd):
        """
        计算优势和回报
        :param values: 价值模型的输出，shape: [batch_size, num_actions]
        :param rewards: 实际奖励，shape: [batch_size, num_actions]
        :param action_mask: 动作掩码，shape: [batch_size, num_actions]
        :param gamma: 折扣因子
        :param lambd: GAE参数
        :return: 优势和回报，shape: [batch_size, num_actions]
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)  # 获取num_actions的长度

        if action_mask is not None:
            values = values * action_mask
            rewards = rewards * action_mask
        
        # 从后往前根据递推公式计算优势
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0   # 获取 V(t+1)
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]         # 计算 R(t) + gamma * V(t+1) - V(t)
            lastgaelam = delta + gamma * lambd * lastgaelam                   # 计算 [R(t) + gamma * V(t+1) - V(t)] + gamma * lambd * A(t+1)
            advantages_reversed.append(lastgaelam)
        
        # PPO这里的优势是分布在每个token上的，而GRPO是一个句子只有一个优势值
        advantages = torch.stack(advantages_reversed[::-1], dim=1)     # shape: [batch_size, num_actions]
        returns = advantages + values

        return advantages, returns

    # 生成经验
    def generate_experiences(self, samples_list):
        self.actor_model.eval()
        self.ref_model.eval()
        self.reward_model.eval()
        self.critic_model.eval()

        experiences = []

        for samples in samples_list:
            seqs = samples.seqs
            attention_mask = samples.attention_mask
            action_mask = samples.action_mask
            num_actions = samples.num_actions

            # 计算动作的log概率
            with torch.no_grad():
                # 计算策略模型输出token的概率
                action_log_probs = self.get_action_log_probs(self.actor_model, seqs, attention_mask, num_actions)
                # 计算参考模型输出token的概率
                ref_action_log_probs = self.get_action_log_probs(self.ref_model, seqs, attention_mask, num_actions)

                # 计算价值
                value = self.critic_model(seqs, attention_mask=attention_mask, num_actions=num_actions)
                # 转换为文本
                seq_texts = self.actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
                # 计算奖励
                reward_model_inputs = self.reward_tokenizer(seq_texts, return_tensors='pt', padding=True)
                rewards_score = self.reward_model(**reward_model_inputs.to(self.args.device)).logits      # 奖励模型的输出，相当于最后生成一个token的奖励（结果奖励模型）

                # 计算KL散度
                kl = self.compute_approx_kl(
                    action_log_probs,
                    ref_action_log_probs,
                    action_mask=action_mask
                ).to(self.args.device)

                # 计算实际奖励（叠加了kl散度的值）
                rewards = self.compute_rewards(kl, rewards_score, action_mask, kl_ctl=self.args.kl_ctl, clip_reward_value=self.args.clip_reward_value)

                # 计算优势和回报
                advantages, returns = self.get_advantages_and_returns(value, rewards, action_mask, gamma=self.args.gamma, lambd=self.args.lambd)
        
            # ！！！注意：生成的经验数据例如advantages、returns等都是detach()的，防止梯度回传到生成的样本上
            experiences.append(Experience(seqs,
                                        action_log_probs.detach(),
                                        value.detach(),
                                        returns.detach(),
                                        advantages.detach(),
                                        attention_mask,
                                        action_mask,
                                        rewards_score.detach(),
                                        samples.response_length,
                                        samples.total_length,
                                        num_actions,
                                        kl.detach(),
                            ))
        return experiences

    ############################################################################
    # 计算策略损失
    def compute_policy_loss(self, log_probs, old_log_probs, advantages, action_mask=None):
        ratio = (log_probs - old_log_probs).exp()  # 计算概率比率
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)    # 取最小值，防止过拟合

        if action_mask is None:
            return loss.mean(-1).mean()
        return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

    # 计算价值损失
    def compute_value_loss(self, values, old_values, returns, action_mask=None, clip_eps = None):
        if clip_eps is None:
            loss = (values - returns).pow(2)
        else:
            # 计算裁剪后的价值损失
            clipped_values = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
            loss = torch.max((values - returns).pow(2), (clipped_values - returns).pow(2))
        
        if action_mask is None:
            return loss.mean(-1).mean()
        return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()
        

    # 对一个batch的经验数据进行一轮训练
    def train_step(self, experience, steps):
        # 读取一个batch下的经验数据
        sequences = experience.seqs
        old_action_log_probs = experience.action_log_probs        # 真实进行采样的action_log_probs
        advantages = experience.advantages
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        action_mask = experience.action_mask
        old_values = experience.values
        returns = experience.returns

        # 更新actor模型
        self.actor_model.train()
        self.optimizer_actor.zero_grad()
        # 重新计算当前actor_model输出概率分布，如果迭代次数self.args.num_iterations为1，则等价于old_action_log_probs
        action_log_probs = self.get_action_log_probs(self.actor_model, input_ids=sequences, attention_mask=attention_mask, num_actions=num_actions)
        policy_loss = self.compute_policy_loss(action_log_probs, old_action_log_probs, advantages, action_mask=action_mask)
        policy_loss.backward()
        self.optimizer_actor.step()
        self.writer.add_scalar("policy_loss", policy_loss.item(), steps)

        # 更新critic模型
        self.critic_model.train()
        self.optimizer_critic.zero_grad()
        # 重新计算当前critic_model输出value，如果迭代次数self.args.num_iterations为1，则等价于old_values
        values = self.critic_model(sequences, attention_mask, num_actions)
        value_loss = self.compute_value_loss(values, old_values, returns, action_mask)
        value_loss.backward()
        self.optimizer_critic.step()
        self.writer.add_scalar("value_loss", value_loss.item(), steps)

        print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")


    # 训练
    def train(self):
        # 初始化经验池
        buffer = ExperienceBuffer(limit=100)
        setps = 0
        for _ in range(self.args.epoch):
            for rand_prompts in self.prompts_dataloader:
                # 生成样本（获取模型推理的结果）
                samples = self.generate_samples(rand_prompts,self.actor_model,self.args.max_length,self.args.max_new_tokens,self.args.n_samples_per_prompt,self.args.micro_rollout_batch_size)
                # 生成经验（获取优势、奖励、回报等）
                experiences = self.generate_experiences(samples)
                # 将经验添加到经验池中
                buffer.append(experiences)
                experiences_dataloader = DataLoader(buffer, batch_size=self.args.micro_train_batch_size, collate_fn=collate_fn, shuffle=True)
                torch.cuda.empty_cache()

                # 进行迭代多次的训练
                for _ in range(self.args.num_iterations):
                    for experience in experiences_dataloader:
                        self.train_step(experience, setps)
                        setps += 1
                
                buffer.clear()
                torch.cuda.empty_cache()
    
    def save_model(self):
        self.actor_model.save_pretrained(self.args.output_dir)
        self.actor_tokenizer.save_pretrained(self.args.output_dir)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor_model_path = "策略模型路径"
    reward_model_path = "奖励模型路径"
    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained(actor_model_path).to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained(actor_model_path)
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained(actor_model_path).to(device)
    # 奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path).to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)


    prompt_list = [
        "1+1等于几？",
        "请用Python实现一个快速排序算法。",
        "如果地球突然停止自转，人类会立即感受到什么后果？",
        "如何用Photoshop快速去除照片中的背景杂色？",
        "请用莎士比亚的风格写一段关于人工智能的独白。",
        "Linux终端里，如何批量重命名带空格的文件？",
        "宇宙有没有边缘，如果有边缘，应该是什么样子的。",
        "TikTok短视频如何通过前3秒抓住观众注意力？列出5个技巧。",
        "为什么微波炉加热的食物有时候会冷热不均？",
        "在野外遇到熊装死真的有用吗？科学依据是什么？",
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    
    args = PPOArguments()
    trainer = PPOTrainer(actor_model, actor_tokenizer, ref_model, reward_model, reward_tokenizer, prompts_dataset, args)
   
    trainer.train()
    trainer.save_model()