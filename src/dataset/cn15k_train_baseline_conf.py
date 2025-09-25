import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets


model_name = 'sbert'
path = 'dataset/ukg/train_50neighbor/confidence_prediction/cn15k_0.85train'
path_wo_conf = 'dataset/ukg/train_50neighbor/cn15k_0.85train'

train_df = pd.read_json(f'{path}/train_1hop_conffilter.json')
val_df = pd.read_json(f'{path}/val_1hop_conffilter.json')
test_df = pd.read_json(f'{path}/test_1hop_conffilter.json')
dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

path_nodes = f'{path_wo_conf}/nodes'
path_edges = f'{path_wo_conf}/edges'
path_graphs= f'{path_wo_conf}/graphs'




class CN15kBaselineConfDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.\nAnswer:'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        self.dataset = dataset

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        question = f'Question: What is the probability that the fact "{data["triple"].lower()}" is true?\nOnly output a number between 0 and 1 without any explanation.\nAnswer: '
        graph = torch.load(f'{path_graphs}/{index}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        # desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

        # 创建 node_id 到 node_attr 的映射
        node_id_to_attr = dict(zip(nodes['node_id'], nodes['node_attr']))
        # 将 edges 中的 src 和 dst 替换为对应的 node_attr
        edges['src'] = edges['src'].map(node_id_to_attr)
        edges['dst'] = edges['dst'].map(node_id_to_attr)

        # 随机选择 50 行，如果行数不足 50 则选择全部行
        sampled_edges = edges.sample(n=min(50, len(edges)), random_state=42)
        desc = sampled_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

        if isinstance(data['confidence'], list):
            label = ('|').join(data['confidence'])
        else:
            label = data['confidence']

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path_wo_conf}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path_wo_conf}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path_wo_conf}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = CN15kBaselineConfDataset()

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
