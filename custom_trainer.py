import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, Tuple, Union


class CustomTrainer(Trainer):
    def __init__(self, *args, kl_loss_weight=0.5, ce_loss_weight=1.0, mse_loss_weight=0.5, true_token_id, false_token_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_loss_weight = kl_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        print(f"True token id: {self.true_token_id}")
        print(f"False token id: {self.false_token_id}")
        print(f"ce_loss_weight: {self.ce_loss_weight}")
        print(f"kl_loss_weight: {self.kl_loss_weight}")
        print(f"mse_loss_weight: {self.mse_loss_weight}")

        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
        self.mse_criterion = nn.MSELoss(reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False):
        confidence = inputs.pop("confidence", None)
        labels = inputs.get("labels", None)
        outputs = model(**inputs)
        original_loss = outputs.loss
        if confidence is not None:
            kl_loss, mse_loss = self.compute_first_token_loss(model, outputs, confidence, labels)
            total_loss = self.ce_loss_weight * original_loss + self.kl_loss_weight * kl_loss + self.mse_loss_weight * mse_loss
            if self.state.global_step % 50 == 0:
                print(f"Step {self.state.global_step} - Original loss: {original_loss.item():.4f}, "
                      f"KL loss: {kl_loss.item():.4f}, MSE loss: {mse_loss.item():.4f}, Total loss: {total_loss.item():.4f}")
        else:
            total_loss = original_loss
        return (total_loss, outputs) if return_outputs else total_loss



    def compute_first_token_loss(self, model, outputs, confidence, labels):
        """
        计算第一个 token 的 KL 散度损失和 MSE 损失。
        Args:
            outputs: model outputs, including logits
            confidence: shape: (batch_size,)
            labels: shape: (batch_size, seq_len)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (kl_loss, mse_loss)
        """
        # outputs.logits shape: (batch_size, seq_len, vocab_size)
        logits = outputs.logits
        batch_size = logits.shape[0]
        vocab_size = logits.shape[2]
        
        # 1. 找到每个样本中第一个有效的label位置（即第一个输出token）
        IGNORE_INDEX = -100
        first_token_positions = []
        for i in range(batch_size):
            # .nonzero(as_tuple=True)[0] 返回非零元素的索引张量
            valid_positions = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0] 
            if len(valid_positions) > 0:
                first_token_positions.append(valid_positions[0].item())
            else:
                first_token_positions.append(-1)

        # 2. 确定有效的样本索引
        valid_samples_indices = [i for i, pos in enumerate(first_token_positions) if pos >= 0]
        
        if len(valid_samples_indices) == 0:
            zero = torch.tensor(0.0, device=logits.device)
            return zero, zero # 返回零 KL 损失和零 MSE 损失
        
        # 3. 提取第一个输出 token 的 Logits, Confidence, 和 Target Distribution
        first_token_logits = torch.stack([
            logits[i, first_token_positions[i]] 
            for i in valid_samples_indices
        ])  # shape: (num_valid_samples, vocab_size)
        
        if hasattr(model, "module"):
            calibration_head = model.module.calibration_head
        else:
            calibration_head = model.calibration_head
        target_dtype = first_token_logits.dtype
        # 3. 显式地将 Calibration Head 的权重移动到目标数据类型
        calibration_head = calibration_head.to(target_dtype)
        # 将原始 logits 通过 FFN 映射
        calibrated_logits = calibration_head(first_token_logits)

        # 提取有效样本的 confidence
        valid_confidence = confidence[valid_samples_indices] # shape: (num_valid_samples,)
        valid_confidence = valid_confidence.to(dtype=logits.dtype)

        # 构建目标概率分布 (仅用于 KLDivLoss)
        target_dist = torch.zeros(len(valid_samples_indices), vocab_size, device=logits.device, dtype=logits.dtype)
        target_dist[:, self.true_token_id] = valid_confidence
        target_dist[:, self.false_token_id] = 1.0 - valid_confidence
        
        # 4. 计算 KL 散度损失 (D_KL(P || Q))
        log_probs = F.log_softmax(calibrated_logits, dim=-1)
        kl_loss = self.kl_criterion(log_probs, target_dist)
        
        # 5. 计算 MSE 损失
        # 提取 true_token_id 对应的概率
        # P(true_token | first_token_logits)
        true_token_probs = F.softmax(calibrated_logits, dim=-1)[:, self.true_token_id] # shape: (num_valid_samples,)
        # MSE(P(true_token) - confidence)
        # valid_confidence 相当于目标值，true_token_probs 相当于预测值
        mse_loss = self.mse_criterion(true_token_probs, valid_confidence)
        
        return kl_loss, mse_loss
