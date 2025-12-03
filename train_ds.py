import os
import wandb
import gc
from tqdm import tqdm
import time
import json
import pandas as pd
import sys
from typing import Any, Dict, List
import concurrent.futures

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset
from torch_geometric.data import Batch

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

print("start import from src/ ...")
start = time.time()
from src.dataset import load_dataset
from src.model import load_model, llama_model_path
from src.utils.evaluate import eval_funcs
from src.utils.ckpt import _save_checkpoint, _reload_best_model, _save_checkpoint_nooptim
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
from src.config import parse_args_llama
# from src.model.pt_llm_ds import PromptTuningLLM
# from src.model.graph_llm_ds import GraphLLM
print(f"end import from src! cost {(time.time()-start):.1f}s")


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100



class GraphTextCollator(DataCollatorForSeq2Seq):
    """
    一个自定义的 collator，用于同时处理文本（由 DataCollatorForSeq2Seq 处理）
    和图数据（由 PyG 的 Batch.from_data_list 处理）。
    """

    # **添加 __init__ 方法来接收 fp16 参数**
    def __init__(self, tokenizer, model=None, padding=True, pad_to_multiple_of=None, return_tensors=None, fp16: bool = False):
        # 调用父类的构造函数，传入所有必要的参数
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        # **将 fp16 保存为实例属性**
        self.fp16 = fp16
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # 1. 从 features 中分离 graph 和 text
        graph_list = []
        text_features = []
        graph_path_list = []

        for feature in features:
            # 弹出 'graph'，这样父类 collator 就不会处理它
            # feature.pop() 会返回 'graph' 键对应的值，并从字典中移除它
            graph_data = feature.pop("graph", None)
            if graph_data is not None:
                graph_list.append(graph_data)
            
            graph_path_data = feature.pop("graph_path", None)
            if graph_path_data is not None:
                graph_path_list.append(graph_path_data)
            
            # 剩下的键（input_ids, labels 等）被添加到 text_features
            text_features.append(feature)

        # 2. 使用父类 DataCollatorForSeq2Seq 来处理所有文本相关的键
        # 这将自动完成 padding
        batch = super().__call__(text_features, return_tensors=self.return_tensors)

        # 3. 如果存在图数据，使用 PyG 的 Batch 来打包它们
        if graph_list:
            # from_data_list 会将图列表打包成一个单一的、
            # 包含不相连子图的大图对象 (Batch)
            batched_graph = Batch.from_data_list(graph_list)

            if self.fp16:
                # 在执行计算前，我们希望输入数据的类型与模型权重（float16）匹配。
                # 注意：如果您的环境只支持 BF16，则需要改为 torch.bfloat16
                target_dtype = torch.float16
                
                # 转换节点特征 (x) 的数据类型
                # 这是一个 CPU 到 GPU 的传输，但更重要的是类型转换
                if hasattr(batched_graph, 'x') and batched_graph.x is not None:
                    # 确保转换类型
                    batched_graph.x = batched_graph.x.to(target_dtype)
                    
                # 如果您的图数据还有其他张量特征（如边特征 'edge_attr'），也应该一并转换
                if hasattr(batched_graph, 'edge_attr') and batched_graph.edge_attr is not None:
                    if batched_graph.edge_attr.is_floating_point():
                        batched_graph.edge_attr = batched_graph.edge_attr.to(target_dtype)
            # 将打包好的图添加回 batch 字典
            batch['graph'] = batched_graph
        
        if graph_path_list:
            batch['graph_path'] = graph_path_list
        return batch



class GpuCacheClearingCallback(TrainerCallback):
    """
    一个自定义的 Callback，用于在每个 epoch 结束时清除 GPU 缓存。
    """
    def on_epoch_end(self, **kwargs):
        # 仅在 GPU 可用时执行清理操作
        if torch.cuda.is_available():
            print("\n*** cuda.empty_cache... ***")
            # 1. 强制 Python 垃圾回收
            gc.collect()
            # 2. 清除 PyTorch 的 GPU 缓存
            torch.cuda.empty_cache()
            # 可选：打印当前的显存使用情况（仅供调试）
            # print(f"当前分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            # print(f"当前缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            print("*** end cuda.empty_cache ***\n")




def main(args):
    # Step 1: Set up wandb
    seed = args.seed
    # wandb.init(project=f"{args.project}",
    #            name=f"{args.dataset}_{args.model_name}_seed{seed}",
    #            config=args)
    seed_everything(seed=args.seed)
    print(args)


    # Step 2: Build Model
    print('start load model...')
    start = time.time()

    args.llm_model_path = llama_model_path[args.llm_model_name]

    tokenizer = LlamaTokenizer.from_pretrained(args.llm_model_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    print(f'world_size: {world_size}, ddp: {ddp}')
    if ddp:  
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}   # distributed data parallel
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    else:  
        device_map = "auto"   # model parallel
    print(f'device_map: {device_map}')

    args.grad_steps = gradient_accumulation_steps


    model = LlamaForCausalLM.from_pretrained(
        args.llm_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        max_memory={i: f'{size}GiB' for i, size in enumerate(args.max_memory)},
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
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
        model.print_trainable_parameters()
    

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()


    sp_model = load_model[args.model_name](model=model, tokenizer=tokenizer, graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)
    # sp_model = GraphLLM(model=model, tokenizer=tokenizer, graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)
    print(f'end load model! cost {(time.time()-start):.1f}s')
    trainable_params, all_param = sp_model.print_trainable_params()
    print(f"sp_model trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")







    def generate_and_tokenize_prompt(data_point):
        questions = tokenizer(data_point["question"], add_special_tokens=False)
        descriptions = tokenizer(data_point["desc"], add_special_tokens=False)
        labels = tokenizer(data_point["label"], add_special_tokens=False)

        eos_tokens = tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)
        # bos_tokens = tokenizer(BOS, add_special_tokens=False)

        label_input_ids = labels.input_ids[:args.max_new_tokens] + eos_tokens.input_ids
        # input_ids = bos_tokens.input_ids + descriptions.input_ids[:args.max_txt_len] + questions.input_ids + eos_user_tokens.input_ids + label_input_ids
        input_ids = descriptions.input_ids[:args.max_txt_len] + questions.input_ids + eos_user_tokens.input_ids + label_input_ids

        label_input_ids = [IGNORE_INDEX] * (len(input_ids) - len(label_input_ids)) + label_input_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids],
            "labels": label_input_ids,
            "graph": data_point["graph"],
            # "graph_path": data_point["graph_path"],
        }


    # Step 3: Build  Dataset
    print('start load dataset...')
    start = time.time()

    try:
        train_dataset, val_dataset = dataset.load_train_val_data_from_pickle()
    except:
        train_dataset = [dataset[i] for i in idx_split['train']]
        val_dataset = [dataset[i] for i in idx_split['val']]
    train_dataset = [generate_and_tokenize_prompt(i) for i in tqdm(train_dataset, desc="train dataset tokenize...")]
    val_dataset = [generate_and_tokenize_prompt(i) for i in tqdm(val_dataset, desc="val dataset tokenize...")]

    try:
        test_dataset = dataset.load_test_data_from_pickle()
    except:
        test_dataset = Subset(dataset, idx_split['test'])
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=8)
    print(f'end load dataset! cost {(time.time()-start):.1f}s')









    # Step 5. Training
    print('start training...')

    # args.micro_batch_size = args.batch_size // args.grad_steps
    output_path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}_fp16_{args.fp16}/'
    args.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    print(f'output_path: {output_path}')
    print(f'batch_size: {args.batch_size}, micro_batch_size: {args.micro_batch_size}, grad_steps: {args.grad_steps}, world_size: {world_size}')

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_steps,
        learning_rate=args.lr,
        weight_decay=args.wd,
        logging_strategy="steps",
        logging_steps=args.grad_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="wandb",
        fp16=args.fp16,
        optim="adamw_torch",
        # warmup_steps=100,
        warmup_ratio=0.2,
        lr_scheduler_type="cosine",
        deepspeed=args.deepspeed,
    )

    # 创建自定义 collator 的实例
    data_collator = GraphTextCollator(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=sp_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )



    model.config.use_cache = False
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    _save_checkpoint_nooptim(sp_model, args)





    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 5. Evaluating
    print('start testing...')
    path = f'{output_path}test_result.csv'
    print(f'path: {path}')

    # sp_model = _reload_best_model(sp_model, args)
    sp_model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                output = sp_model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    # Step 6. Post-processing & compute metrics
    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')
    # wandb.log({'Test Acc': acc})


if __name__ == "__main__":
    print('start load args ...')
    start = time.time()
    args = parse_args_llama()
    print(f'end load args! cost {(time.time()-start):.1f}s')

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
