from collections import defaultdict
from scipy.sparse import lil_matrix
import os
import json
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from torch_sparse import tensor

def get_data(features = 64):
    train_file_path = os.path.join(os.getcwd(), 'mixed.txt')

    with open(train_file_path, encoding='utf-8') as f:
        train_json_data = json.load(f)

    paper_ids = set(str(paper['publication_ID']) for paper in train_json_data)
    paper_citation_list = [(str(paper['publication_ID']), 
                            [cit for cit in str(paper['Citations']).split(';') if cit in paper_ids])
                        for paper in train_json_data if 'Citations' in paper]
    
    # create ids set and give them index
    unique_ids = paper_ids
    id_to_index = {pid: idx for idx, pid in enumerate(unique_ids)}
    index_to_id = {idx: pid for pid, idx in id_to_index.items()}

    paper_citation_indexed = [(id_to_index[str(pc[0])], [id_to_index[cit] for cit in pc[1] if cit in id_to_index]) 
                          for pc in paper_citation_list if str(pc[0]) in id_to_index]
    
    num_papers = len(id_to_index)
    adj_matrix = lil_matrix((num_papers, num_papers))

    index_to_data = {id_to_index[str(paper['publication_ID'])]: paper 
                 for paper in train_json_data if 'publication_ID' in paper}
    
    for row_idx, col_indices in paper_citation_indexed:
        for col_idx in col_indices:
            adj_matrix[row_idx, col_idx] = 1

    row, col = adj_matrix.nonzero()
    edge_index = torch.stack((torch.tensor(row), torch.tensor(col)), dim=0)
    edge_index = edge_index.to(torch.long)
    adj_tensor = tensor.SparseTensor(row=edge_index[0], 
                                    col=edge_index[1], 
                                    sparse_sizes=(num_papers, num_papers))
    
    paper_texts = [
        (str(paper['title']) + ' ' + str(paper['abstract']) + ' ' + (' '.join(str(keyword) for keyword in paper['keywords']) if isinstance(paper['keywords'], (list, tuple)) else '')).lower()
        for paper in train_json_data
        if 'publication_ID' in paper and 'title' in paper and 'abstract' in paper and 'keywords' in paper
    ]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=features)
    node_features = vectorizer.fit_transform(paper_texts)

    # convert to dense tensor
    node_features = node_features.toarray()

    # create node embeddings
    node_embeddings = np.zeros((len(paper_ids), node_features.shape[1]))
    for pid, idx in id_to_index.items():
        node_embeddings[idx] = node_features[idx]

    # create node years
    node_years = np.zeros((len(paper_ids), 1))
    for paper in train_json_data:
        if 'publication_ID' in paper and 'pubDate' in paper:
            pid = str(paper['publication_ID'])
            if pid in id_to_index:
                idx = id_to_index[pid]
                year = int(paper['pubDate'][:4])  # extract year
                node_years[idx] = year

    return adj_tensor, torch.tensor(node_embeddings, dtype=torch.float), torch.tensor(node_years, dtype=torch.float), edge_index, index_to_data, index_to_id, vectorizer, unique_ids, id_to_index