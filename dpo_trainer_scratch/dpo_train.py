from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dpo_dataset import DPODataset, DPODataCollator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# 是将模型输出的结果转化为概率分布，并且只获取当模型输出为labels下标的概率分布结果
def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)    # 将vocab_size维度归一化为概率分布
    # 在vocab_size维度提取 如果输出为labels下标的结果
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


# 去除padding部分的logits，并且队log概率求和，得到概率乘积的log
def mask_logits(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels_masks shape: (batch_size, seq_len)
    # new_logits shape: (batch_size, 1)
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))
    
    return new_logits


# dpo_loss的计算
def dpo_loss(ref_probs, probs, beta):
    # batch前半部分是chosen，后半部分是rejected，需要切分开
    def split_probs(probs):
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)
    
    # 将chosen和rejected分开
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)

    ref_logratios = ref_chosen_probs - ref_reject_probs
    pi_logratios = chosen_probs - reject_probs

    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta*logits)
    return loss.mean()


class DPOTrainer(Trainer):
    def __init__(
        self, ref_model=None, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        
        # 关键：确保 ref_model 不参与梯度计算
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.args.device)  # 同步到训练设备
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    # 重写loss计算函数
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        labels = inputs['labels']

        # 获取ref_model输出的logits
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=input_ids, labels = labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)   # (batch_size, seq_len)
        ref_probs = mask_logits(ref_probs, labels)        # (batch_size, 1)，将每个token的概率分布转化为概率乘积

        logits = model(input_ids=input_ids, labels = labels).logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)

        loss = dpo_loss(ref_probs, probs, 0.1)
        return loss
    

if __name__ == "__main__":
    mode_path = "模型路径"
    model = AutoModelForCausalLM.from_pretrained(mode_path)
    ref_model = AutoModelForCausalLM.from_pretrained(mode_path).eval()
    
    tokenizer = AutoTokenizer.from_pretrained("mode_path", use_fast=True)
    data_collator = DPODataCollator(tokenizer, max_seq_len=1024) # 加载的大模型旋转位置编码最大长度为1024，这里不能超过这个值
    args = TrainingArguments(output_dir='./results/dpo-1-epoch', 
                            num_train_epochs=1,  # 训练太多轮，模型似乎会输出很多重复内容
                            do_train=True, 
                            per_device_train_batch_size=16,
                            gradient_accumulation_steps=4,
                            # max_steps=15000,
                            logging_steps=50,
                            report_to='tensorboard',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001,  # 学习率很重要，太大会把模型训飞
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            save_steps=100)          
    dataset = DPODataset('./data/dpo.jsonl', tokenizer=tokenizer)
    trainer = DPOTrainer(model=model,
                        ref_model=ref_model,  # 确保 ref_model 被正确传递
                        args=args,
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/dpo-1-epoch')
    trainer.save_state()