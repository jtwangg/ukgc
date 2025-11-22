import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch_geometric.data import Batch

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphLLMDSCT(PreTrainedModel):

    def __init__(
        self,
        model,
        tokenizer,
        args,
        **kwargs
    ):
        super(GraphLLMDSCT, self).__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

        self.true_token_id = self.tokenizer.encode("True", add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode("False", add_special_tokens=False)[0]
        print(f"True token id: {self.true_token_id}, False token id: {self.false_token_id}")

        vocab_size = self.model.config.vocab_size

        self.if_calibration = args.if_calibration
        if self.if_calibration:
            self.calibration_head = nn.Sequential(
                nn.Linear(vocab_size, vocab_size // 2),  
                nn.GELU(), 
                nn.Linear(vocab_size // 2, vocab_size) 
            ).to(self.model.device)

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

    def encode_graphs(self, graphs):
        graphs = graphs.to(self.model.device)
        # print(f'graphs.x.dtype: {graphs.x.dtype}')
        # print(f"self.graph_encoder.lin1.weight dtype: {self.graph_encoder.convs[0].lin_query.weight.dtype}")
        n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)

        # mean pooling
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')

        return g_embeds
    
    # def encode_graphs_inference(self, samples):
    #     graphs = samples['graph']
    #     graphs = graphs.to(self.model.device)
    #     n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)
    #     # mean pooling
    #     g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
    #     return g_embeds

    def forward(self, input_ids, attention_mask, labels, graph=None, graph_path=None, confidence=None):
        batch_size, seq_len = input_ids.shape
        token_embeds = self.word_embedding(input_ids)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device)).unsqueeze(0).repeat(batch_size, 1, 1).to(token_embeds.device)
        
        # encode graph
        if graph is not None:
            graph_embeds = self.encode_graphs(graph)
        elif graph_path is not None:
            graph_list = []
            for gp in graph_path:
                graph = torch.load(gp)
                graph_list.append(graph)
            batched_graph = Batch.from_data_list(graph_list)
            # 类型转换，for 混合精度训练
            target_dtype = self.graph_encoder.convs[0].lin_query.weight.dtype
            if hasattr(batched_graph, 'x') and batched_graph.x is not None:
                batched_graph.x = batched_graph.x.to(target_dtype)
            if hasattr(batched_graph, 'edge_attr') and batched_graph.edge_attr is not None:
                if batched_graph.edge_attr.is_floating_point():
                    batched_graph.edge_attr = batched_graph.edge_attr.to(target_dtype)
            graph_embeds = self.encode_graphs(batched_graph)


        graph_embeds = self.projector(graph_embeds)
        graph_embeds = graph_embeds.unsqueeze(1).to(token_embeds.device)
        # print(f"Shape of bos_embeds: {bos_embeds.shape}")
        # print(f"Shape of graph_embeds: {graph_embeds.shape}")
        # print(f"Shape of token_embeds: {token_embeds.shape}")
        input_embeds = torch.cat((bos_embeds, graph_embeds, token_embeds), dim=1)
        soft_prompt_length = bos_embeds.shape[1] + graph_embeds.shape[1]
        new_attention_mask = torch.cat((torch.ones((batch_size, soft_prompt_length), device=attention_mask.device), attention_mask), dim=1)
        new_labels = torch.cat((torch.full((batch_size, soft_prompt_length), IGNORE_INDEX, dtype=labels.dtype, device=labels.device), labels), dim=1)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=new_attention_mask,
                return_dict=True,
                labels=new_labels,
            )

        return outputs


    def inference_customloss(self, samples):
        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        if samples.get("graph") is not None:
            batched_graph = samples["graph"]
            target_dtype = self.graph_encoder.convs[0].lin_query.weight.dtype
            if hasattr(batched_graph, 'x') and batched_graph.x is not None:
                batched_graph.x = batched_graph.x.to(target_dtype)
            if hasattr(batched_graph, 'edge_attr') and batched_graph.edge_attr is not None:
                if batched_graph.edge_attr.is_floating_point():
                    batched_graph.edge_attr = batched_graph.edge_attr.to(target_dtype)
            graph_embeds = self.encode_graphs(batched_graph)
        elif samples.get("graph_path") is not None:
            graph_list = []
            for gp in samples["graph_path"]:
                graph = torch.load(gp)
                graph_list.append(graph)
            batched_graph = Batch.from_data_list(graph_list)
            # 类型转换，for 混合精度训练
            target_dtype = self.graph_encoder.convs[0].lin_query.weight.dtype
            if hasattr(batched_graph, 'x') and batched_graph.x is not None:
                batched_graph.x = batched_graph.x.to(target_dtype)
            if hasattr(batched_graph, 'edge_attr') and batched_graph.edge_attr is not None:
                if batched_graph.edge_attr.is_floating_point():
                    batched_graph.edge_attr = batched_graph.edge_attr.to(target_dtype)
            graph_embeds = self.encode_graphs(batched_graph)
        # graph_embeds = self.encode_graphs_inference(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            first_token_logits = outputs.logits[:, -1, :]

            if self.if_calibration:
                calibrated_logits = self.calibration_head(first_token_logits)
            else:
                calibrated_logits = first_token_logits

            all_token_probs = torch.softmax(calibrated_logits, dim=-1) # shape: (batch_size, vocab_size)
            # 直接取出 True 和 False 对应的概率
            true_probs = all_token_probs[:, self.true_token_id].cpu().tolist() # shape: (batch_size, 1)
            false_probs = all_token_probs[:, self.false_token_id].cpu().tolist()

            # 然后进行正常的生成
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True
            )
        
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return {
            'id': samples['id'],
            'pred': pred,
            'label': samples['label'],
            'confidence': samples['confidence'],
            'true_prob': true_probs,
            'false_prob': false_probs,
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
