import math
import contextlib
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class PromptTuningLLM(PreTrainedModel):

    def __init__(
        self,
        model,
        init_prompt,
        args,
        **kwargs
    ):
        super(PromptTuningLLM, self).__init__(model.config)
        self.model = model
        num_virtual_tokens = args.llm_num_virtual_tokens

        self.word_embedding = self.model.model.get_input_embeddings()

        # prompt tuning
        init_token_ids = self.tokenizer(init_prompt).input_ids
        num_text_tokens = len(init_token_ids)
        if num_text_tokens < num_virtual_tokens:
            num_reps = math.ceil(num_virtual_tokens / num_text_tokens)
            init_token_ids = init_token_ids * num_reps
        init_token_ids = init_token_ids[:num_virtual_tokens]

        self.prompt = torch.nn.Parameter(self.word_embedding.weight[torch.LongTensor(init_token_ids)].detach().clone().to(torch.float32)).to(self.model.device)

    # @property
    # def device(self):
    #     return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, input_ids, attention_mask, labels):

        # # encode description, questions and labels
        # questions = self.tokenizer(samples['question'], add_special_tokens=False)
        # descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        # labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # # encode special tokens
        # eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        # eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        # bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        # pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # batch_size = len(samples['id'])
        # batch_inputs_embeds = []
        # batch_attention_mask = []
        # batch_label_input_ids = []
        # prompt_embeds = self.prompt.repeat(batch_size, 1)
        # for i in range(batch_size):
        #     # Add bos & eos token
        #     label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
        #     input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
        #     inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
        #     inputs_embeds = torch.cat([bos_embeds, prompt_embeds, inputs_embeds], dim=0)

        #     batch_inputs_embeds.append(inputs_embeds)
        #     batch_attention_mask.append([1] * inputs_embeds.shape[0])
        #     label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
        #     batch_label_input_ids.append(label_input_ids)

        # # pad inputs_embeds
        # max_length = max([x.shape[0] for x in batch_inputs_embeds])
        # for i in range(batch_size):
        #     pad_length = max_length-batch_inputs_embeds[i].shape[0]
        #     batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
        #     batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
        #     batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        # inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        # attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        # label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        batch_size, seq_len = input_ids.shape
        token_embeds = self.word_embedding(input_ids)
        prompt_embeds = self.prompt.repeat(batch_size, 1)
        input_embeds = torch.cat((prompt_embeds, token_embeds), dim=1)
        prompt_length = prompt_embeds.shape[1]
        new_attention_mask = torch.cat((
            torch.ones((batch_size, prompt_length), device=attention_mask.device),
            attention_mask
        ), dim=1)
        # 创建新的标签，prompt部分设为IGNORE_INDEX（表示忽略），原始输入部分使用传入的labels
        new_labels = torch.cat((
            torch.full((batch_size, prompt_length), IGNORE_INDEX, dtype=labels.dtype, device=labels.device),
            labels
        ), dim=1)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=new_attention_mask,
                return_dict=True,
                labels=new_labels,
            )

        return outputs.loss

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
        prompt_embeds = self.prompt.repeat(batch_size, 1)
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, prompt_embeds, inputs_embeds], dim=0)
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
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
