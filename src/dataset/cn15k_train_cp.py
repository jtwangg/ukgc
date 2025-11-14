import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst, retrieval_via_pcst_graphweight
import random
from io import StringIO
import pickle

model_name = 'sbert'
path = 'dataset/ukg/train_50neighbor/confidence_prediction_cleantraingraph/cn15k'

train_df = pd.read_json(f'{path}/train_1hop.json')
val_df = pd.read_json(f'{path}/val_1hop.json')
test_df = pd.read_json(f'{path}/test_1hop.json')
dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

CACHE_DIR = f'{path}/cached_graphs_pkl'

def preprocess():
    """
    从graph中检索一个子图，保存到cached_graph和cached_desc中
    """
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)

    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        graph = torch.load(f'{path_graphs}/{index}.pt')
        q_emb = q_embs[index]
        subg, desc = retrieval_via_pcst_graphweight(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)



class CN15kCPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        self.dataset = dataset
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        prompt设计
        """
        data = self.dataset.iloc[index]
        question = f'Question: What is the probability that the fact "{data["question"].lower()}" is true? Only output a number between 0 and 1 without any explanation.\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()  # description

        # 从cached_graph/xx.pt读取desc内容
        content = open(f'{cached_desc}/{index}.txt', 'r').read()  # description
        # 分割节点和边的内容
        node_content, edge_content = content.split('\n\n')[0], content.split('\n\n')[1]
        # 创建节点的 DataFrame
        node_df = pd.read_csv(StringIO(node_content))
        # 创建 node_id 到 node_attr 的映射
        node_id_to_attr = dict(zip(node_df['node_id'], node_df['node_attr']))
        # 创建边的 DataFrame
        edge_df = pd.read_csv(StringIO(edge_content))
        # 将 edges 中的 src 和 dst 替换为对应的 node_attr
        edge_df['src'] = edge_df['src'].map(node_id_to_attr)
        edge_df['dst'] = edge_df['dst'].map(node_id_to_attr)
        # 生成 desc 字符串
        desc = edge_df.to_csv(index=False, columns=['src', 'edge_attr', 'dst', 'weight'])

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

    def load_train_val_data_from_pickle(self):
        with open(f"{CACHE_DIR}/train.pkl", 'rb') as f:
            train = pickle.load(f)
        with open(f"{CACHE_DIR}/val.pkl", 'rb') as f:
            val = pickle.load(f)
        return train, val

    def load_test_data_from_pickle(self):
        with open(f"{CACHE_DIR}/test.pkl", 'rb') as f:
            test = pickle.load(f)
        return test
        

def cache_data():
    dataset = CN15kCPDataset()
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
    print(f'cache dir: {CACHE_DIR}')
    for split_name, data_list in tqdm(data_to_save.items(), desc="保存缓存文件"):
        file_path = f"{CACHE_DIR}/{split_name}.pkl"
        with open(file_path, 'wb') as f:
            # 使用 pickle 序列化并保存数据
            pickle.dump(data_list, f)
        print(f"成功保存 {len(data_list)} 个 {split_name} 项目到 {file_path}")


if __name__ == '__main__':

    # preprocess()

    # dataset = CN15kCPDataset()

    # idx = 101
    # data = dataset[idx]
    # for k, v in data.items():
    #     print(f'{k}: {v}')

    # split_ids = dataset.get_idx_split()
    # for k, v in split_ids.items():
    #     print(f'# {k}: {len(v)}')

    cache_data()