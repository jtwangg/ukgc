import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd

from tqdm import tqdm
from torch_geometric.data.data import Data

from src.utils.lm_modeling import load_model, load_text2embedding
import pandas as pd



model_name = 'sbert'
path = '/data/jtwang/G-Retriever/dataset/ukg/ppi5k_0.7train'

train_df = pd.read_json(f'{path}/train_1hop_conffilter.json')
val_df = pd.read_json(f'{path}/val_1hop_conffilter.json')
test_df = pd.read_json(f'{path}/test_1hop_conffilter.json')
dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

def step_one():
    """
    生成nodes文件夹和edges文件夹，存储每个graph的节点和边的csv文件
    """

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        nodes = {}
        edges = []
        row = dataset.iloc[i]
        for tri in row['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges.to_csv(f'{path_edges}/{i}.csv', index=False)


def step_two():
    """
    对问题query和图节点和边使用sbert进行编码，保存向量pt文件
    """
    print('Loading dataset...')
    questions = [row['question'].lower() for i, row in dataset.iterrows()]
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f'{path}/q_embs.pt')

    # encode graphs
    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for index in tqdm(range(len(dataset))):
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        nodes.node_attr.fillna("", inplace=True)
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        # encode nodes
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # encode edges
        # edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())  # 会重复编码

        # 建立 edge_attr 到 edge_id 的映射
        unique_edge_attrs = edges.edge_attr.unique().tolist()
        edge_attr_to_id = {attr: idx for idx, attr in enumerate(unique_edge_attrs)}
        edge_ids = [edge_attr_to_id[attr] for attr in edges.edge_attr.tolist()]
        # 对唯一的 edge_attr 进行编码
        unique_edge_embeddings = text2embedding(model, tokenizer, device, unique_edge_attrs)
        # 根据映射关系获取每个边对应的编码
        edge_attr = unique_edge_embeddings[edge_ids]
        
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])  # head id & tail id
        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes)) # nodes向量，[head列表, tail列表]，edge向量，节点数量
        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')



def generate_split():
    """
    生成train, val, test的索引，保存到split文件夹下
    """
    train_indices = np.arange(len(train_df))
    val_indices = np.arange(len(val_df)) + len(train_df)
    test_indices = np.arange(len(test_df)) + len(train_df) + len(val_df)

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))

if __name__ == '__main__':
    step_one()
    step_two()
    generate_split(len(dataset), f'{path}/split', 0.999, 0.999)



