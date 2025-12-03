import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
import pickle
import time
from tqdm import tqdm
import os

model_name = 'sbert'
path = 'dataset/ukg/train_50neighbor/confidence_prediction_cleantraingraph/nl27k'

train_df = pd.read_json(f'{path}/train_1hop.json')
val_df = pd.read_json(f'{path}/val_1hop.json')
test_df = pd.read_json(f'{path}/test_1hop.json')
dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs= f'{path}/graphs'




class NL27kBaselineTCDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.\nAnswer:'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        self.dataset = dataset

        self.all_graphs = torch.load(f'{path}/all_graphs.pt')
        with open(f'{path}/all_edges.pkl', 'rb') as f:
            self.all_edges = pickle.load(f)
        with open(f'{path}/all_nodes.pkl', 'rb') as f:
            self.all_nodes = pickle.load(f)

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        question = f'Question: Determine the correctness of the fact "{data["question"].lower()}".\nAnswer by "True" or "False" without any explanation.\nAnswer: '
        # graph = torch.load(f'{path_graphs}/{index}.pt')
        # nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        # edges = pd.read_csv(f'{path_edges}/{index}.csv')
        graph = self.all_graphs[index]
        nodes = self.all_nodes[index]
        edges = self.all_edges[index]
        # desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

        # 创建 node_id 到 node_attr 的映射
        node_id_to_attr = dict(zip(nodes['node_id'], nodes['node_attr']))
        # 将 edges 中的 src 和 dst 替换为对应的 node_attr
        edges['src'] = edges['src'].map(node_id_to_attr)
        edges['dst'] = edges['dst'].map(node_id_to_attr)

        # 将权重weight转为true/false
        # edges['weight'] = (edges['weight'].astype(float) >= 0.85).astype(str)

        # 随机选择 50 行，如果行数不足 50 则选择全部行
        # sampled_edges = edges.sample(n=min(50, len(edges)), random_state=42)
        desc = edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst', 'weight'])

        label = (data['answer'].astype(float) >= 0.5).astype(str)
        confidence = data['answer'].astype(float)

        return {
            'id': index,
            'question': question,
            'label': label,
            'confidence': confidence,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def get_max_desc_length():
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("/seu_share/home/qiguilin/220236147/huggingface_models/Llama-2-7b-chat-hf")
    dataset = NL27kBaselineTCDataset()
    idx_split = dataset.get_idx_split()
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]
    train_dataset_desc_length = [len(tokenizer(i["desc"], add_special_tokens=False).input_ids) for i in train_dataset]
    val_dataset_desc_length = [len(tokenizer(i["desc"], add_special_tokens=False).input_ids) for i in val_dataset]
    test_dataset_desc_length = [len(tokenizer(i["desc"], add_special_tokens=False).input_ids) for i in test_dataset]
    print(f'train dataset max length: {max(train_dataset_desc_length)}')
    print(f'val dataset max length: {max(val_dataset_desc_length)}')
    print(f'test dataset max length: {max(test_dataset_desc_length)}')


if __name__ == '__main__':
    dataset = NL27kBaselineTCDataset()

    data = dataset[100]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')

    # get_max_desc_length()
