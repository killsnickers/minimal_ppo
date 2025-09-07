"""
PPO Trainer minimal
From KIMI 2.0
@豌杂 2025-09-07
"""
# ppo_full_template.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import os
import math
import time
import json


# ---------- 超参 ----------
MODEL_ID = "qwen3/0.6B"  # 762M
MAX_LEN = 512
BATCH_SIZE = 1          # 梯度累积模拟大batch
GRAD_ACCUM = 16
LR = 1e-5
EPOCHS = 1
PPO_EPOCHS = 4
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
GAMMA = 0.99
LAMBDA = 0.95
MAX_GRAD_NORM = 1.0
DEVICE = "cuda"

# ---------- 数据 ----------
class SimpleDataset(Dataset):
    def __init__(self, tokenizer, n=1000):
        self.tokenizer = tokenizer
        self.pairs = [
            ("What is 2+2?", "4"),
            ("Who are you?", "I am your AI assistant."),
            ("Explain gravity.", "Gravity attracts masses."),
        ] * (n//3 + 1)
        self.pairs = self.pairs[:n]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        prompt = f"User: {q}\nAssistant:"
        full = f"{prompt} {a}"
        tok = self.tokenizer(full, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return tok.input_ids[0]

# ---------- 模型 ----------
class ActorCritic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        hidden = model.config.n_embd
        self.v_head = nn.Linear(hidden, 1)
        self.ref_model = AutoModelForCausalLM.from_config(model.config).to(DEVICE)
        self.ref_model.load_state_dict(self.model.state_dict())
        for p in self.ref_model.parameters(): p.requires_grad = False
    def forward(self, input_ids, attention_mask=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = out.logits
        v = self.v_head(out.hidden_states[-1]).squeeze(-1)
        return logits, v
    def ref_logits(self, input_ids):
        with torch.no_grad():
            return self.ref_model(input_ids=input_ids).logits

# ---------- 采样 ----------
@torch.no_grad()
def sample(model_actor_critic, tokenizer, prompt, max_new=64):
    model_actor_critic.eval()
    tok = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = tok.input_ids
    gen = model_actor_critic.model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response, gen

# ---------- 奖励 ----------
def reward_fn(prompt, response):
    # 简单规则：包含数字给1，否则0
    return 1.0 if any(ch.isdigit() for ch in response) else 0.0

# ---------- GAE ----------
def compute_gae(rewards, values, next_value, gamma=GAMMA, lam=LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] - values[step]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[step])
    return returns

# ---------- PPO step ----------
def ppo_step(model_actor_critic, optimizer, tokenizer, prompts, responses, old_logprobs, returns, advantages):
    model_actor_critic.train()
    total_loss = 0
    for _ in range(PPO_EPOCHS):
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            batch_responses = responses[i:i+BATCH_SIZE]
            # 拼接 prompt+response
            texts = [p+r for p,r in zip(batch_prompts, batch_responses)]
            tok = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            input_ids = tok.input_ids
            logits, values = model_actor_critic(input_ids, tok.attention_mask)
            values = values[:, :-1]
            logits = logits[:, :-1, :]
            # logprob
            shift_labels = input_ids[:, 1:]
            logprob = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1), reduction='none')
            logprob = logprob.reshape_as(shift_labels)
            # 只取response部分
            prompt_lens = [len(tokenizer.encode(p)) for p in batch_prompts]
            response_logprobs = []
            for j, pl in enumerate(prompt_lens):
                response_logprobs.append(logprob[j, pl-1:-1].sum())
            logprob = torch.stack(response_logprobs)
            # ratio
            ratio = torch.exp(logprob - old_logprobs[i:i+BATCH_SIZE])
            # clip
            clip_adv = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advantages[i:i+BATCH_SIZE]
            policy_loss = -torch.min(ratio * advantages[i:i+BATCH_SIZE], clip_adv).mean()
            # value loss
            value_pred = values.mean(dim=1)
            vf_loss = F.mse_loss(value_pred, returns[i:i+BATCH_SIZE])
            # entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs+1e-8)).sum(-1).mean()
            loss = policy_loss + VF_COEF*vf_loss - ENT_COEF*entropy
            # backward
            (loss / GRAD_ACCUM).backward()
            if (i+1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model_actor_critic.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
    return total_loss / (len(prompts)//BATCH_SIZE*PPO_EPOCHS)

# ---------- 主流程 ----------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    model_ac = ActorCritic(base_model).to(DEVICE)
    optimizer = torch.optim.AdamW(model_ac.parameters(), lr=LR)

    dataset = SimpleDataset(tokenizer, n=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(EPOCHS):
        prompts, responses, old_logprobs, rewards, values = [], [], [], [], []
        for batch in dataloader:
            input_ids = batch.to(DEVICE)
            prompt_ids = input_ids[:, :input_ids.shape[1]//2]  # 前一半当prompt
            prompt = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            response, gen = sample(model_ac, tokenizer, prompt)
            r = reward_fn(prompt, response)
            # 记录
            prompts.append(prompt)
            responses.append(response)
            # 计算old_logprob
            full_tok = tokenizer(prompt+response, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model_ac.ref_logits(full_tok.input_ids)
            shift_logits = logits[:, :-1, :]
            shift_labels = full_tok.input_ids[:, 1:]
            logprob = -F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), reduction='none')
            logprob = logprob.reshape_as(shift_labels)
            prompt_len = len(tokenizer.encode(prompt))
            response_logprob = logprob[0, prompt_len-1:-1].sum()
            old_logprobs.append(response_logprob)
            rewards.append(r)
            # value
            with torch.no_grad():
                _, v = model_ac(full_tok.input_ids)
                values.append(v[0, -1].item())
        # 计算returns & advantages
        next_value = 0
        returns = compute_gae(rewards, values, next_value)
        advantages = [ret - val for ret, val in zip(returns, values)]
        # 转tensor
        old_logprobs = torch.stack(old_logprobs).to(DEVICE)
        returns = torch.tensor(returns).to(DEVICE)
        advantages = torch.tensor(advantages).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std()+1e-8)
        # PPO update
        loss = ppo_step(model_ac, optimizer, tokenizer, prompts, responses, old_logprobs, returns, advantages)
        print(f"Epoch {epoch} loss={loss:.4f}")

    # 保存
    model_ac.model.save_pretrained("ppo_full_1b")
    tokenizer.save_pretrained("ppo_full_1b")

if __name__ == "__main__":
    main()