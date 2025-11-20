import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, Tuple, Union


class CustomTrainer(Trainer):
    def __init__(self, *args, custom_loss_weight, ce_loss_weight, true_token_id, false_token_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_weight = custom_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        print(f"True token id: {self.true_token_id}")
        print(f"False token id: {self.false_token_id}")
        print(f"Custom loss weight: {self.custom_loss_weight}")

    def compute_loss(self, model, inputs, return_outputs=False):
        confidence = inputs.pop("confidence", None)
        labels = inputs.get("labels", None)
        outputs = model(**inputs)
        original_loss = outputs.loss
        if confidence is not None and self.custom_loss_weight > 0:
            custom_loss = self.compute_first_token_loss(outputs, confidence, labels)
            total_loss = self.ce_loss_weight * original_loss + self.custom_loss_weight * custom_loss
            if self.state.global_step % 100 == 0:
                print(f"Step {self.state.global_step} - Original loss: {original_loss.item():.4f}, "
                      f"Custom loss: {custom_loss.item():.4f}, Total loss: {total_loss.item():.4f}")
        else:
            total_loss = original_loss
        return (total_loss, outputs) if return_outputs else total_loss



    def compute_first_token_loss(self, outputs, confidence, labels):
        """
        compute first token loss
        Args:
            outputs: model outputs, including logits
            confidence: shape: (batch_size,
        Returns:
            custom loss
        """
        # outputs.logits shape: (batch_size, seq_len, vocab_size)
        logits = outputs.logits
        batch_size = logits.shape[0]
        vocab_size = logits.shape[2]
        
        # 找到每个样本中第一个有效的label位置（即第一个输出token）
        IGNORE_INDEX = -100
        first_token_positions = []
        for i in range(batch_size):
            valid_positions = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                first_token_positions.append(valid_positions[0].item())
            else:
                # 如果没有有效位置，使用-1标记
                first_token_positions.append(-1)
        
        # 构建目标概率分布
        # target_dist shape: (batch_size, vocab_size)
        target_dist = torch.zeros(batch_size, vocab_size, device=logits.device, dtype=logits.dtype)
        
        valid_samples = []
        for i in range(batch_size):
            pos = first_token_positions[i]
            if pos >= 0:
                conf = confidence[i].item()
                target_dist[i, self.true_token_id] = conf
                target_dist[i, self.false_token_id] = 1.0 - conf
                valid_samples.append(i)
        
        if len(valid_samples) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # 提取第一个输出token的logits
        first_token_logits = torch.stack([
            logits[i, first_token_positions[i]] 
            for i in valid_samples
        ])  # shape: (num_valid_samples, vocab_size)
        
        valid_target_dist = target_dist[valid_samples]  # shape: (num_valid_samples, vocab_size)
        
        # 计算交叉熵损失, 使用log_softmax + 手动计算交叉熵
        # log_probs = F.log_softmax(first_token_logits, dim=-1)
        # custom_loss = -(valid_target_dist * log_probs).sum(dim=-1).mean()

        # --- 使用 torch.nn.KLDivLoss 替换手动交叉熵 ---
        # 1. 计算 Log-Softmax (得到模型输出的 log-probabilities: log(Q))
        log_probs = F.log_softmax(first_token_logits, dim=-1)
        # 2. 初始化 KLDivLoss 
        # reduction='batchmean' 确保了损失值是所有样本的平均值，与您原来的 .mean() 行为一致
        kl_criterion = nn.KLDivLoss(reduction='batchmean')
        # 3. 计算 KL 散度损失 D_KL(P || Q)
        # KLDivLoss(log(Q), P)
        custom_loss = kl_criterion(log_probs, valid_target_dist)
        
        return custom_loss
