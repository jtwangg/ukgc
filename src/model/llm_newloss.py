import contextlib
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class LLM_newloss(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            "max_memory": {i: f'{size}GiB' for i, size in enumerate(args.max_memory)},
            "device_map": "auto",
            "revision": "main",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)

            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        self.word_embedding = self.model.model.get_input_embeddings()
        
        # 可学习的温度系数，初始化为1.0
        self.temperature = torch.nn.Parameter(torch.ones(1))
        
        # 保存损失权重
        self.original_loss_weight = 1.0
        self.custom_loss_weight = 0.1

        # 解析 True / False 的词ID，避免硬编码
        true_id = self.tokenizer.convert_tokens_to_ids("True")
        false_id = self.tokenizer.convert_tokens_to_ids("False")
        # 回退方案：如果未能直接转换，则使用编码的最后一个子词ID
        if true_id is None or true_id == 0 and hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None and true_id == self.tokenizer.unk_token_id:
            encoded_true = self.tokenizer("True", add_special_tokens=False).input_ids
            true_id = encoded_true[-1] if len(encoded_true) > 0 else 0
        if false_id is None or false_id == 0 and hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None and false_id == self.tokenizer.unk_token_id:
            encoded_false = self.tokenizer("False", add_special_tokens=False).input_ids
            false_id = encoded_false[-1] if len(encoded_false) > 0 else 0
        self.true_token_id = true_id
        self.false_token_id = false_id

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples):
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        
        # 解析 confidence 字段，将字符串转换为 float tensor
        confidences = torch.tensor([float(conf) for conf in samples["confidence"]], 
                                   dtype=torch.float32, device=self.model.device)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        batch_generation_positions = []  # 记录每个样本的生成起始位置

        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            
            # 记录生成位置（label 开始的位置）
            generation_pos = inputs_embeds.shape[0] - len(label_input_ids)
            batch_generation_positions.append(generation_pos)
            
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]
            batch_generation_positions[i] = batch_generation_positions[i] + pad_length  # 调整位置

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)
        generation_positions = torch.tensor(batch_generation_positions, device=self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,  # 保留原始设置，自动计算原始损失
            )

        # 获取原始损失（包含完整序列和 EOS token）
        original_loss = outputs.loss
        
        # 提取生成位置的 logits
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 为每个样本提取第一个生成位置的 logits
        batch_indices = torch.arange(batch_size, device=self.model.device)
        generation_logits = logits[batch_indices, generation_positions]  # [batch_size, vocab_size]
        
        # 提取二分类目标（"True" 与 "False"）对应的 logits，并应用温度系数
        true_false_logits = generation_logits[:, [self.true_token_id, self.false_token_id]] / self.temperature  # [batch_size, 2]
        
        # 计算对数概率
        # 在 float32 中计算以提升数值稳定性
        log_probs = torch.log_softmax(true_false_logits.float(), dim=-1)  # [batch_size, 2]
        
        # 构建目标分布 [confidence, 1-confidence]
        target_probs = torch.stack([confidences, 1 - confidences], dim=1)  # [batch_size, 2]
        
        # 计算自定义置信度损失
        custom_loss = -torch.sum(target_probs * log_probs, dim=1).mean()
        
        # 混合损失
        loss = self.original_loss_weight * original_loss + self.custom_loss_weight * custom_loss

        return loss

    def inference(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            if len(descriptions.input_ids[i]) > self.max_txt_len:
                print(f'description too long! length: {len(descriptions.input_ids[i])}')
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True,  # IMPORTANT!
                output_scores=True,
                return_dict_in_generate=True,
            )

        # 首步 logits -> 概率 -> 取 True 的概率
        first_step_scores = outputs.scores[0]
        probs = torch.softmax(first_step_scores, dim=-1)
        pred_confidence = probs[:, self.true_token_id].detach().float().cpu().tolist()

        # 解码生成序列
        pred = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'pred_confidence': pred_confidence,
                'label': samples['label'],
                'confidence': samples['confidence'],
                'question': samples['question'],
                'desc': samples['desc'],
                 }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
