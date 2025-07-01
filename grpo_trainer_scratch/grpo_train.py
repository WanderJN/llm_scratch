from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from copy import deepcopy
import torch
from datasets import load_dataset
from reward_func import *

class GSM8KDataset:
    def __init__(self, data_path: str, tokenizer: AutoTokenizer):
        self.data = load_dataset(data_path)["train"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        prompt = item["question_zh-cn"]
        answer = item["answer_only"]
        return {
            "prompt": prompt,
            "answer": answer
        }

# 存储一次采样的数据
@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.LongTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int

@dataclass
class GRPOArguments:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./output"
    lr = 1e-6
    save_steps = 100
    epochs = 3
    batch_size = 1
    num_iterations = 2    # 每次采样的迭代次数
    num_generations = 4   # 每次采样的组内样本数
    gradient_accumulation_steps = 2  # 梯度累积步数
    max_prompt_length = 512
    max_generate_length = 512
    reward_weight : List[float] = None
    beta = 0.0            # KL散度的系数，如果为0则不使用KL散度
    clip_eps = 0.2

class GRPOTrainer:
    def __init__(self, model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 reward_funcs: Union[List[str], List[Callable]] = None,
                 args: GRPOArguments = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 reward_tokenizers = None):
        self.args = args

        os.makedirs(args.output_dir, exist_ok=True)  # 创建输出文件夹

        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            tokenizer.padding_side = "left"

        self.tokenizer = tokenizer

        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            # 如果奖励函数是字符串，则加载预训练的模型
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        
        self.reward_funcs = reward_funcs

        # 加载奖励模型的分词器
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(self.reward_funcs)
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")
        # 如果分词器没加载好，需要手动加载
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        
        self.reward_tokenizers = reward_tokenizers

        # 加载其他配置信息
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # 缓存已经生成的数据的一个批次的数据，可以供模型多次迭代训练，无需重新采样生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps

        # 记录模型参数更新的次数
        self.update_steps = 0
    

    # GRPO的采样生成样本，以组为单位
    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs["prompt"]]
        answers = [None] * len(prompts)

        if "answer" in inputs:
            answers = [answer for answer in inputs["answer"]]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)
            
            # 生成一个group组的输入数据
            inputs = self.tokenizer([input_text] * self.args.num_generations, padding="max_length", max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')

            # 对一组数据进行推理
            with torch.no_grad():
                prompt_response_ids = self.model.generate(**inputs.to(self.args.device),
                                                        max_new_tokens = self.args.max_generate_length,
                                                        temperature=0.9,
                                                        top_p = 1,
                                                        top_k = 50)
            # 对推理结果进行处理，如果超过最大输入+输出长度，就截断
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
        
            # 整理ids和attention_mask
            prompt_ids = inputs["input_ids"]
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)


            # 将一个group的采样结果保存下来
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt = prompt,
                answer = answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)
        
        return samples_list
    
    # 生成经验(优势、token的概率分布)
    def generate_experience(self, inputs: Union[Dataset, List[Samples]]) -> List[Samples]:
        self.model.eval()
        samples_list = self.generate_samples(inputs)

        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []

        # 遍历每个group的样本，计算rewards和优势
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len)
            response_ids = samples.response_ids # shape: (num_generations, seq_len)
            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len)
            action_mask = samples.action_mask # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)

            with torch.no_grad():
                # 计算策略模型输出token的概率
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)

                # 是否使用参考模型
                if self.ref_model is not None:
                    # 计算参考模型输出token的概率
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                # 存储各奖励函数在一个group里的相应值
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)

                # 将输出转化为文本，把prompt和response分开，用于传入奖励函数或奖励模型，计算奖励
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]

                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    # 如果是奖励模型，传入prompt和response的文本
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    # 如果是奖励函数，传入prompt和response的文本以及答案
                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                
                # 汇总计算全部奖励函数的结果
                # 如果没有设置奖励权重，则默认每个奖励函数的权重为1.0
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                
                 # 乘以各个奖励函数的权重，rewards_per_func: [num_funcs, num_generations]
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)

                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                print(f'rewards: {rewards}')
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # 计算最终的优势值，GRPO的优势是句子粒度的，而非token粒度的
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                batch_advantages.append(advantages)
        
               
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }
    
    def compute_loss(self, model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        prompt_response_ids = inputs["prompt_response_ids"]
        attention_mask = inputs["attention_mask"]
        action_mask = inputs["action_mask"]
        num_actions = action_mask.size(1)
        # 重新计算当前输出概率分布，如果迭代次数self.args.num_iterations为1，则等价于old_action_log_probs
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        # 如果使用参考模型，则需要计算kl散度
        if self.args.beta != 0.0:
            ref_action_log_probs = inputs["ref_action_log_probs"]
            log_ratio = ref_action_log_probs - action_log_probs
            log_ratio = log_ratio * action_mask

            # GRPO的KL散度计算公式是k3
            k3 = log_ratio.exp() - 1 - log_ratio

        advantages = inputs["advantages"]
        old_action_log_probs = inputs["old_action_log_probs"] if self.args.num_iterations > 1 else action_log_probs.detach()   # 之前的动作的log概率

        # 计算GRPO的loss
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # 一个序列中每个token的优势是一样的
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask     # shape: [batch_size, num_generations, seq_len]

        # 加上kl散度的loss计算
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # 先在token上平均loss
        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)  # shape: [batch_size * num_generations]
        # 再在batch上平均loss
        loss = loss.mean()

        return loss

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

    
    def train_step(self, model, inputs, optimizer, step):
        model.train()

        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps  # 梯度累积

        loss.backward()

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()


    def train(self):
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            
            for idx, batch in enumerate(dataloader):
                # 生成经验和优势
                inputs = self.generate_experience(batch)
                # 将inputs存入缓存，经验可以重复利用self.args.num_iterations次
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs

                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # 一次经验，多次训练
                    for _ in range(self.args.num_iterations):
                        # 每次迭代都从缓存中取出数据
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1

                        # 达到检查步数就保存模型
                        if self.update_steps % self.args.save_steps == 0:
                            checkpoint_file = self.args.output_dir + f'/checkpoint_{self.update_steps}'
                            os.makedirs(checkpoint_file, exist_ok=True)

                            self.model.save_pretrained(checkpoint_file)
                            self.tokenizer.save_pretrained(checkpoint_file)
                del inputs
    
    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""
    args = GRPOArguments()

    writer = SummaryWriter('./runs')

    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    
    
    prompts_dataset = GSM8KDataset('/home/user/Downloads/gsm8k_chinese', tokenizer)
  
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()