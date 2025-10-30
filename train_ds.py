import os
import wandb
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import time
from torch.utils.data import Subset
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100


class DataCollatorForCausalLM(DataCollatorWithPadding):
    def __init__(self, tokenizer, max_txt_len, max_new_tokens):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.max_txt_len = max_txt_len
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
    
    def __call__(self, original_batch):
        batch = {}
        for k in original_batch[0].keys():
            batch[k] = [d[k] for d in original_batch]
        if 'graph' in batch:
            batch['graph'] = Batch.from_data_list(batch['graph'])

        # encode description, questions and labels
        questions = self.tokenizer(batch["question"], add_special_tokens=False)
        descriptions = self.tokenizer(batch["desc"], add_special_tokens=False)
        labels = self.tokenizer(batch["label"], add_special_tokens=False)

        # encode sepcial tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_tokens = self.tokenizer(BOS, add_special_tokens=False)

        batch_size = len(batch['id'])
        batch_input_ids = []
        batch_label_input_ids = []

        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_token.input_ids
            input_ids = bos_tokens.input_ids + descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            batch_input_ids.append(input_ids)
            label_input_ids = [IGNORE_INDEX] * (len(input_ids) - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        
        max_length = max([len(x) for x in batch_input_ids])
        for i in range(batch_size):
            pad_length = max_length - len(batch_input_ids[i])
            batch_input_ids[i] = [self.tokenizer.pad_token_id] * pad_length + batch_input_ids[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([[1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in ids] for ids in batch_input_ids], dtype=torch.long),
            "labels": torch.tensor(batch_label_input_ids, dtype=torch.long),
        }



def main(args):
    print("start import from src/ ...")
    start = time.time()
    from src.dataset import load_dataset
    from src.model import load_model, llama_model_path
    from src.utils.evaluate import eval_funcs
    
    from src.utils.ckpt import _save_checkpoint, _reload_best_model
    from src.utils.collate import collate_fn
    from src.utils.seed import seed_everything
    from src.utils.lr_schedule import adjust_learning_rate
    print(f"end import from src! cost {(time.time()-start):.1f}s")
    
    # Step 1: Set up wandb
    seed = args.seed
    # wandb.init(project=f"{args.project}",
    #            name=f"{args.dataset}_{args.model_name}_seed{seed}",
    #            config=args)
    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()


    # Step 3: Build Model
    print('start load model...')
    start = time.time()
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)
    print(f'end load model! cost {(time.time()-start):.1f}s')

    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(
    #     [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
    #     betas=(0.9, 0.95)
    # )
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")






    # Step 2: Build Node Classification Dataset
    print('start load dataset...')
    start = time.time()
    # train_dataset = [dataset[i] for i in idx_split['train']]
    # val_dataset = [dataset[i] for i in idx_split['val']]
    # test_dataset = [dataset[i] for i in idx_split['test']]
    train_dataset = Subset(dataset, idx_split['train'])
    val_dataset = Subset(dataset, idx_split['val'])
    test_dataset = Subset(dataset, idx_split['test'])
    print(f'end load dataset! cost {(time.time()-start):.1f}s')

    # Initialize custom data collator
    data_collator = DataCollatorForCausalLM(tokenizer=model.tokenizer, max_txt_len=args.max_txt_len, max_new_tokens=args.max_new_tokens)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=8)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=data_collator, num_workers=8)
    print(f'end load dataset! cost {(time.time()-start):.1f}s')







    args.micro_batch_size = args.batch_size // args.grad_steps

    
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}'
    os.makedirs(path, exist_ok=True)
    print(f'path: {path}')


    training_args = TrainingArguments(
        output_dir=path,
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
        report_to=None,
        fp16=True,
        optim="adamw_hf",
        warmup_steps=100,
        deepspeed=args.deepspeed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )


    # Step 5. Training
    print('start training...')

    trainer.train()



    # num_training_steps = args.num_epochs * len(train_loader)
    # progress_bar = tqdm(range(num_training_steps))
    # best_val_loss = float('inf')

    # for epoch in range(args.num_epochs):
    #     model.train()
    #     epoch_loss, accum_loss = 0., 0.

    #     for step, batch in enumerate(train_loader):

    #         optimizer.zero_grad()
    #         loss = model(batch)
    #         loss.backward()

    #         clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

    #         if (step + 1) % args.grad_steps == 0:
    #             adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

    #         optimizer.step()
    #         epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

    #         if (step + 1) % args.grad_steps == 0:
    #             lr = optimizer.param_groups[0]["lr"]
    #             # wandb.log({'Lr': lr})
    #             # wandb.log({'Accum Loss': accum_loss / args.grad_steps})
    #             print(f'Lr: {lr}, Accum Loss: {accum_loss / args.grad_steps}')
    #             accum_loss = 0.

    #         progress_bar.update(1)

    #     print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
    #     # wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

    #     val_loss = 0.
    #     eval_output = []
    #     model.eval()
    #     with torch.no_grad():
    #         for step, batch in enumerate(val_loader):
    #             loss = model(batch)
    #             val_loss += loss.item()
    #         val_loss = val_loss/len(val_loader)
    #         print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
    #         # wandb.log({'Val Loss': val_loss})

    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         _save_checkpoint(model, optimizer, epoch, args, is_best=True)
    #         best_epoch = epoch

    #     print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

    #     if epoch - best_epoch >= args.patience:
    #         print(f'Early stop at epoch {epoch}')
    #         break



    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 5. Evaluating
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'path: {path}')


    # model = _reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
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
    from src.config import parse_args_llama
    args = parse_args_llama()
    print(f'end load args! cost {(time.time()-start):.1f}s')

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
