"""
PPO Trainer minimal
依葫芦画瓢做的模型，这里还有很多逻辑，主要是先跑起来看看
补充：
这里我们明确几个概念
return:
value:
reward:
@豌杂 2025-09-07
"""
# ppo_full.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import os
import math
import time
import json


# ----基础的参数设置---
MODEL_NAME = 'QWEN3/0.6B'
SAVE_DIR = ''

MAX_LENGTH = 1024  # 输入最大长度
MAX_NEW_TOKEN_LENGTH = 500
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2

LR = 1e-6
EPOCH = 3
PPO_EPOCHS = 4  # 一次采样，迭代多少个epoch
CLIP_EPS = 0.2
VF_COEF = 0.5   # value func的系数
ENTROPY_COEF = 0.01  # 熵的系数

GAMMA = 0.99  # 不知道什么含义
LAMBDA = 0.95  # 不知道什么含义
MAX_GRAD_NORM = 1.0
DEVICE = "cuda"


# ----数据处理函数，继承Dataset ----
class minimal_dataset(Dataset):
    """
    最终实现的get_item, 返回tokenizer后的input_ids
    """
    def __init__(self, tokenizer, max_num=1000):
        self.tokenizer = tokenizer
        self.data = [
            ("What is 2+2?", "4"),
            ("Who are you?", "I am your AI assistant."),
            ("Explain gravity.", "Gravity attracts masses."),
        ] * (max_num//3 + 1)
        self.data = self.data[:max_num]

    def __len__(self): return len(self.data)

    def get_item(self, index):
        q, a = self.data[index]
        prompt = f"User: {q}\nAssistant:"
        full = f"{prompt} {a}"
        # tokenizer, 然后做截断， 截断长度为MAX_LENGTH
        tokens = self.tokenizer(full, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return tokens.input_ids[0]


# ---- Actor和Critic的model计算 ----
class ActorCritic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # value 计算的逻辑
        hidden = model.config.n_embd
        self.value_head = nn.Linear(hidden, 1)

        # ref model load， 不做梯度计算
        self.ref_model = AutoModelForCausalLM.from_config(model.config).to(DEVICE)
        self.ref_model.load_state_dict(self.model.state_dict())
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = output.logits
        value = self.value_head(output.hidden_states[-1]).squeeze(-1)
        return logits, value

    def ref_forward(self, input_ids):
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=input_ids).logits
            return ref_logits


# ---- 采样数据 ----
@torch.no_grad()
def sample(model_actor_critic, tokenizer, prompt):
    model_actor_critic.eval()
    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = tokens.input_ids
    gen = model_actor_critic.model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKEN_LENGTH,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    # 只获取其中的response部分
    response = tokenizer.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response, gen


# ---- 奖励函数 ----
def reward_fn(prompt, response):
    # 简单规则：包含数字给1，否则0
    return 1.0 if any(ch.isdigit() for ch in response) else 0.0


# ---- GAE的计算 ----
# 这里需要详细的了解一下，所谓的GAE到底是什么？为什么加这个？其实这里计算的是每个token的实际return
def compute_gae(rewards, values, next_value, gamma=GAMMA, lam=LAMBDA):
    """
    详细拆解一下这里， rewards 这里表示的是仅仅当前action的reward, value代表的是当前状态下的value function
    那么 当前状态下， 采取Action的优势advantage = reward + value[n+1] - value[n]
    因此这里的advantage 就会是有正有负的， 并且这就是当前的结果
    """
    values = values + [next_value]   # ？这一步还不知道为什么？
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):  # 记住这里有个reverse, 从最后一个token往前算的
        delta = rewards[step] + gamma * values[step+1] - values[step]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[step])
    return returns


# ---- ppo的每一次迭代step ----
def ppo_step(model_actor_critic, optimizer, tokenizer, prompts, responses, old_logprobs, returns, advantages):
    """
    核心的计算、迭代和优化的过程，前置的采样，数据等等都获取之后的
    """
    model_actor_critic.train()
    total_loss = 0
    for _ in range(PPO_EPOCHS):
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_responses = responses[i:i + BATCH_SIZE]
            # 拼接 prompt+response
            texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
            tok = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
            input_ids = tok.input_ids

            # 可以认为是在当前的policy model里面，做了一遍 inference
            logits, values = model_actor_critic(input_ids, tok.attention_mask)
            values = values[:, :-1]   # b * (len-1)
            logits = logits[:, :-1, :]     # b * (len-1) * v

            # policy的logprob
            shift_labels = input_ids[:, 1:]
            logprob = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1), reduction='none')
            logprob = logprob.reshape_as(shift_labels)   # b * (len-1), 表示是取到当前的token的probs

            prompt_lens = [len(tokenizer.encode(p)) for p in batch_prompts]
            response_logprobs = []
            for j, pl in enumerate(prompt_lens):
                response_logprobs.append(logprob[j, pl - 1:-1].sum())
            logprob = torch.stack(response_logprobs)

            # 当前的logprobs - old_logprobs   的  指数， 则就是P(π) / P(ref)
            ratio = torch.exp(logprob - old_logprobs[i:i + BATCH_SIZE])
            # clip 核心逻辑,   ratio、1-clip, 1+clip的中间值， 然后和 ratio * adv的最小值
            # 这里能看到mean()，主要是为了计算loss，loss在这里可以算作是一个标量
            clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[i:i + BATCH_SIZE]
            policy_loss = -torch.min(ratio * advantages[i:i + BATCH_SIZE], clip_adv).mean()

            # value loss， 可以看到， return才是我们真实的回报，而values只是return的预测值
            value_pred = values.mean(dim=1)  # todo：这里的mean 其实还是没看懂， 理论上这里的mean会和后面的returns的size()不太一致
            vf_loss = F.mse_loss(value_pred, returns[i:i + BATCH_SIZE])

            # 这里还计算了熵值
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
            loss = policy_loss + VF_COEF * vf_loss - ENTROPY_COEF * entropy

            (loss / GRADIENT_ACCUMULATION).backward() # 梯度回传
            if (i + 1) % GRADIENT_ACCUMULATION == 0:
                nn.utils.clip_grad_norm_(model_actor_critic.parameters(), MAX_GRAD_NORM) # 做了梯度裁切
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
    return total_loss / (len(prompts) // BATCH_SIZE * PPO_EPOCHS)


# ---- main 函数的实际迭代 ----
def main():
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # load base model(policy model)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model_ac = ActorCritic(base_model).to(DEVICE)

    # 设置优化器optimizes，并且指定LR
    optimizer = torch.optim.AdamW(model_ac.parameters(), lr=LR)

    dataset = minimal_dataset(tokenizer, max_num=256)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH):
        prompts, responses, old_logprobs, rewards, values = [], [], [], [], []
        for batch in dataloader:
            input_ids = batch.to(DEVICE)
            prompt_ids = input_ids[:, :input_ids.shape[1] // 2]  # 前一半当prompt
            prompt = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

            # 提供Prompt, 然后随机采样到所有Response和generate, 并且计算整体的reward
            response, gen = sample(model_ac, tokenizer, prompt)
            reward = reward_fn(prompt, response)

            # 记录
            prompts.append(prompt)
            responses.append(response)

            # 计算old_logprob
            full_tok = tokenizer(prompt + response, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                ref_logits = model_ac.ref_logits(full_tok.input_ids)
            shift_logits = ref_logits[:, :-1, :]   # batch * len * token_num
            shift_labels = full_tok.input_ids[:, 1:]  # batch * len-1
            # shift_logits.reshape -> （b*(len0-1)) * v, shift_labels -> (b*len-1)
            ref_logprob = -F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
                                       reduction='none')
            ref_logprob = ref_logprob.reshape_as(shift_labels)
            prompt_len = len(tokenizer.encode(prompt))
            response_logprob = ref_logprob[0, prompt_len - 1:-1].sum()
            old_logprobs.append(response_logprob)
            rewards.append(reward)
            with torch.no_grad():
                _, v = model_ac(full_tok.input_ids)
                values.append(v[0, -1].item())
        # 对当前的所有数据进行value和基础的信息获取之后，开始计算returns和gae结果了
        next_value = 0
        returns = compute_gae(rewards, values, next_value)
        advantages = [ret - val for ret, val in zip(returns, values)]
        # 转tensor
        old_logprobs = torch.stack(old_logprobs).to(DEVICE)
        returns = torch.tensor(returns).to(DEVICE)
        advantages = torch.tensor(advantages).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss = ppo_step(model_ac, optimizer, tokenizer, prompts, responses, old_logprobs, returns, advantages)
        print(f"Epoch {epoch} loss={loss:.4f}")


    # 保存
    model_ac.model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)


if __name__ == "__main__":
    main()
