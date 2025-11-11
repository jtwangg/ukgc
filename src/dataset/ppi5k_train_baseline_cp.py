import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
import concurrent.futures
import pickle
import os
from tqdm import tqdm


model_name = 'sbert'
path = 'dataset/ukg/train_50neighbor/confidence_prediction_cleantraingraph/ppi5k'

train_df = pd.read_json(f'{path}/train_1hop.json')
val_df = pd.read_json(f'{path}/val_1hop.json')
test_df = pd.read_json(f'{path}/test_1hop.json')
dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs= f'{path}/graphs'


CACHE_DIR = f'{path}/cache'


class PPI5kBaselineCPDataset(Dataset):
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
        question = f'Question: What is the probability that the fact "{data["question"].lower()}" is true?\nOnly output a number between 0 and 1 without any explanation.\nAnswer: '
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
        # sampled_edges = edges.sample(n=min(50, len(edges)), random_state=42)
        desc = edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst', 'weight'])

        label = f"{float(data['answer']):.3f}"

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
            'graph_path': f'{path_graphs}/{index}.pt',
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
    
    def load_data_from_pickle(self):
        with open(f"{CACHE_DIR}/train.pkl", 'rb') as f:
            train = pickle.load(f)
        with open(f"{CACHE_DIR}/val.pkl", 'rb') as f:
            val = pickle.load(f)
        with open(f"{CACHE_DIR}/test.pkl", 'rb') as f:
            test = pickle.load(f)
        return train, val, test


def get_data_item_by_index(dataset, index):
    return dataset[index]


def cache_data():
    dataset = PPI5kBaselineCPDataset()
    idx_split = dataset.get_idx_split()

    train_dataset = [dataset[i] for i in tqdm(idx_split['train'], desc="构建训练集")]
    val_dataset = [dataset[i] for i in tqdm(idx_split['val'], desc="构建验证集")]
    test_dataset = [dataset[i] for i in tqdm(idx_split['test'], desc="构建测试集")]

    # 将数据和对应的文件名后缀存储在一个字典中
    data_to_save = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    # 确保保存目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split_name, data_list in tqdm(data_to_save.items(), desc="保存缓存文件"):
        file_path = f"{CACHE_DIR}/{split_name}.pkl"
        with open(file_path, 'wb') as f:
            # 使用 pickle 序列化并保存数据
            pickle.dump(data_list, f)
        print(f"成功保存 {len(data_list)} 个 {split_name} 项目到 {file_path}")








if __name__ == '__main__':
    # dataset = PPI5kBaselineCPDataset()

    # idx = 101
    # data = dataset[idx]
    # for k, v in data.items():
    #     print(f'{k}: {v}')

    # split_ids = dataset.get_idx_split()
    # for k, v in split_ids.items():
    #     print(f'# {k}: {len(v)}')

    cache_data()