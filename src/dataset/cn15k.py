import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst
import random
from io import StringIO

model_name = 'sbert'
path = 'dataset/ukg/cn15k'
dataset = pd.read_json(f'{path}/test_1hop_conffilter.json')

path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'


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
        subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)



class CN15kDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        self.dataset = pd.read_json(f'{path}/test_1hop_conffilter.json')
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        question = f'Question: {data["question"].lower()} what?\nOnly output your answer without any explanation.\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt')

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
        desc = edge_df.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])




        if isinstance(data['answer'], list):
            label = ('|').join(data['answer']).lower()
        else:
            label = data['answer'].lower()

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
    

if __name__ == '__main__':

    # preprocess()

    dataset = CN15kDataset()

    idx = random.randint(0, len(dataset))
    data = dataset[idx]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')