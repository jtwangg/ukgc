import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
import pickle
import os
import time

start = time.time()
model_name = 'sbert'
path = 'dataset/ukg/train_50neighbor/confidence_prediction_cleantraingraph/nl27k'

train_df = pd.read_json(f'{path}/train_1hop.json')
val_df = pd.read_json(f'{path}/val_1hop.json')
test_df = pd.read_json(f'{path}/test_1hop.json')
dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs= f'{path}/graphs'


CACHE_DIR = f'{path}/graphs_pkl'

class NL27kBaselineCPDataset(Dataset):
    def __init__(self):
        super().__init__()
        start_init = time.time()
        self.prompt = 'Please answer the given question.\nAnswer:'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        self.dataset = dataset

        # 加载所有 graphs
        self.all_graphs = torch.load(f'{path}/all_graphs.pt')
        # 加载所有 nodes
        with open(f'{path}/all_nodes.pkl', 'rb') as f:
            self.all_nodes = pickle.load(f)
        # 加载所有 edges
        with open(f'{path}/all_edges.pkl', 'rb') as f:
            self.all_edges = pickle.load(f)
        end_init = time.time()
        print(f'init cost {(end_init - start_init):.1f} s')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        question = f'Question: What is the probability that the fact "{data["question"].lower()}" is true?\nOnly output a number between 0 and 1 without any explanation.\nAnswer: '
        # graph = torch.load(f'{path_graphs}/{index}.pt')
        # nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        # edges = pd.read_csv(f'{path_edges}/{index}.csv')
        graph = self.all_graphs[index]
        nodes = self.all_nodes[index]
        edges = self.all_edges[index].copy()
        # desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

        # 创建 node_id 到 node_attr 的映射
        node_id_to_attr = dict(zip(nodes['node_id'], nodes['node_attr']))
        # 将 edges 中的 src 和 dst 替换为对应的 node_attr
        edges['src'] = edges['src'].map(node_id_to_attr)
        edges['dst'] = edges['dst'].map(node_id_to_attr)

        # 随机选择 50 行，如果行数不足 50 则选择全部行
        # sampled_edges = edges.sample(n=min(50, len(edges)), random_state=42)
        desc = edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst', 'weight'])

        label = data['answer'].astype(str)

        return {
            'id': index,
            'question': question,
            'label': label,
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
    



def cache_data_per_item():    
    total_items = len(dataset)
    # 用于存储所有数据的列表
    all_graphs = []
    all_nodes = []
    all_edges = []
    
    print(f"Starting to load and collect data for {total_items} items...")
    for index in tqdm(range(total_items), desc="loading..."):
        graph = torch.load(f'{path_graphs}/{index}.pt')
        all_graphs.append(graph)
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        all_nodes.append(nodes)
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        all_edges.append(edges)

    torch.save(all_graphs, f'{path}/all_graphs.pt')
    print(f"All graphs saved to {path}/all_graphs.pt")
    
    with open(f'{path}/all_nodes.pkl', 'wb') as f:
        pickle.dump(all_nodes, f)
    print(f"All nodes saved to {path}/all_nodes.pkl")
    
    with open(f'{path}/all_edges.pkl', 'wb') as f:
        pickle.dump(all_edges, f)
    print(f"All edges saved to {path}/all_edges.pkl")



if __name__ == '__main__':
    # cache_data_per_item()

    dataset = NL27kBaselineCPDataset()

    # idx = 101
    # data = dataset[idx]
    # for k, v in data.items():
    #     print(f'{k}: {v}')

    # split_ids = dataset.get_idx_split()
    # for k, v in split_ids.items():
    #     print(f'# {k}: {len(v)}')

    dataset.get_train()
    end = time.time()
    print(f'cost {(end - start):.1f} s')


