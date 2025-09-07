"""
做了一个最小的CrossEntropy的实现
@豌杂 2025-09-07
"""

import torch
import torch.nn.functional as F   # 仅用来跟官方结果做对比

def my_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                     ignore_index: int = -100, reduction: str = 'mean'):
    """
    极简版交叉熵，等价于
        F.cross_entropy(logits, target, ignore_index=-100, reduction='mean')
    仅支持 2-D logits: (N, C)  和 1-D target: (N,)
    """
    # 0. 基本形状检查
    assert logits.ndim == 2
    assert target.ndim == 1
    N, C = logits.shape
    assert target.shape[0] == N

    # 1. 手动 log-softmax
    # 减去 max 是为了数值稳定
    maxes = logits.max(dim=1, keepdim=True).values
    x = logits - maxes
    logsumexp = x.exp().sum(dim=1, keepdim=True).log()
    log_probs = x - logsumexp          # (N, C)

    # 2. 把 target 变成 0/1 掩码，同时处理 ignore_index
    # 先构造一个 (N, C) 的 0/1 矩阵，只有 target 对应列为 1
    mask = torch.zeros_like(log_probs, dtype=torch.bool)
    valid_mask = target != ignore_index
    valid_target = target.clamp(min=0)          # 负索引会报错，先 clamp
    mask[valid_mask, valid_target[valid_mask]] = 1

    # 3. 交叉熵 = - sum( y * log p ) / N
    loss_vec = -(log_probs * mask).sum(dim=1)   # (N,)
    # 被 ignore 的样本 loss 置 0，不计入后续 reduction
    loss_vec = loss_vec * valid_mask

    if reduction == 'none':
        return loss_vec
    elif reduction == 'sum':
        return loss_vec.sum()
    elif reduction == 'mean':
        return loss_vec.sum() / valid_mask.sum()   # 只均摊有效样本
    else:
        raise ValueError(reduction)


# ------------------ 单元测试 ------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    N, C = 5, 3
    logits = torch.randn(N, C, requires_grad=True)
    target = torch.randint(0, C, (N,))
    target[2] = -100          # 模拟 ignore

    l1 = F.cross_entropy(logits, target, ignore_index=-100, reduction='mean')
    l2 = my_cross_entropy(logits, target)
    print('官方:', l1.item())
    print('自写:', l2.item())
    print('误差:', (l1 - l2).abs().item())