import torch
from torch_geometric.nn import SAGEConv
import numpy as np

import torch.nn.functional as F
import torch_geometric.transforms as T

import data_preprocess

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors

import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define SAGE model
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()
        # store convolution layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # store batchnorm layers
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            # each SAGE conv layer is followed by a corresponding batch normalization layer
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # the last conv transformation
        x = self.convs[-1](x, adj_t)
        return x

def train_mapping_model(x, node_embeddings):
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)
    
    x_np = x.cpu().numpy()
    node_embeddings_np = node_embeddings.detach().cpu().numpy()
    
    model.fit(x_np, node_embeddings_np)
    
    return model

def embed_new_article(title, abstract, keywords, mapping_model, features, vectorizer):
    # concatenate title, abstract, and keywords into a single text
    text = (str(title) + ' ' + str(abstract) + ' ' + (' '.join(str(keyword) for keyword in keywords) if isinstance(keywords, (list, tuple)) else '')).lower()
    
    # use the vectorizer to transform the text into a feature vector
    features = vectorizer.transform([text]).toarray()
    
    # use the mapping model to embed the features into the node embedding space
    embedding = mapping_model.predict(features)
    
    return embedding

def find_nearest_cluster(embedding, cluster_centers):
    # calculate the distance between the new article embedding and each cluster center
    distances = torch.sqrt(torch.sum((embedding - cluster_centers)**2, dim=1))
    
    # find the index of the nearest cluster
    nearest_cluster_index = torch.argmin(distances).item()
    
    return nearest_cluster_index

def get_reference_set(cluster_index, labels, index_to_data, unique_ids):
    # find the indices of articles in the cluster
    article_indices = np.where(labels == cluster_index)[0]
    
    # extract the references of articles in the cluster
    reference_set = set()
    for idx in article_indices:
        article = index_to_data[idx]
        if 'Citations' in article and article['Citations'] is not None:
            references = str(article['Citations']).split(';')
            # print(references)
            filtered_references = [ref for ref in references if ref in unique_ids]
            reference_set.update(filtered_references)
    
    return reference_set

def rank_references(reference_set, index_to_data, id_to_index, labels, node_years, new_article_embedding, node_embeddings, nearest_cluster_index):
    # calculate the similarity between the new article and each reference article
    reference_indices = [id_to_index[ref_id] for ref_id in reference_set if ref_id in id_to_index]
    reference_embeddings = node_embeddings[reference_indices]
    
    nbrs = NearestNeighbors(n_neighbors=len(reference_set), algorithm='ball_tree').fit(reference_embeddings)
    distances, indices = nbrs.kneighbors([new_article_embedding.flatten()])
    
    # calculate the number of times each reference article is cited in the cluster
    reference_counts = {}
    for idx in np.where(labels == nearest_cluster_index)[0]:
        article = index_to_data[idx]
        if 'Citations' in article and article['Citations'] is not None:
            references = str(article['Citations']).split(';')
            for ref in references:
                if ref in reference_set:
                    reference_counts[ref] = reference_counts.get(ref, 0) + 1

    # calculate the total score for each reference article
    reference_scores = []
    for i, ref_index in enumerate(indices[0]):
        ref_data = index_to_data[reference_indices[ref_index]]
        score = 0
        
        # weight of citation 
        count_weight = 0.7
        score += count_weight * reference_counts.get(reference_indices[ref_index], 0)
        
        # weight of similarity
        similarity_weight = 0.1
        score += similarity_weight * (1 - distances[0][i])
        
        # weight of year
        year_weight = 0.2
        year = node_years[reference_indices[ref_index]]
        max_year = torch.max(node_years)
        score += year_weight * (year / max_year)
        
        reference_scores.append((ref_data, score))
    
    # arrange the reference articles in descending order of score
    reference_scores.sort(key=lambda x: x[1], reverse=True)
    
    return reference_scores

def rank_relevant(index_to_data, id_to_index, labels, node_years, new_article_embedding, node_embeddings, nearest_cluster_index, index_to_id):
    # all articles in the cluster
    cluster_indices = np.where(labels == nearest_cluster_index)[0]
    cluster_embeddings = node_embeddings[cluster_indices]
    
    # calculate the similarity between the new article and each article in the cluster
    nbrs = NearestNeighbors(n_neighbors=len(cluster_indices), algorithm='ball_tree').fit(cluster_embeddings)
    distances, indices = nbrs.kneighbors([new_article_embedding.flatten()])
    
    # calculate the number of times each article in the cluster is cited
    citation_counts = {}
    for idx in range(len(index_to_data)):
        article = index_to_data[idx]
        if 'Citations' in article and article['Citations'] is not None:
            citations = str(article['Citations']).split(';')
            for citation in citations:
                if citation in index_to_id:
                    cited_index = id_to_index[citation]
                    if cited_index in cluster_indices:
                        citation_counts[cited_index] = citation_counts.get(cited_index, 0) + 1
    
    # the total score for each article in the cluster
    relevant_scores = []
    for i, article_index in enumerate(indices[0]):
        article_data = index_to_data[cluster_indices[article_index]]
        score = 0
        
        # weight of citation
        count_weight = 0.1
        score += count_weight * citation_counts.get(cluster_indices[article_index], 0)
        
        # weight of similarity
        similarity_weight = 0.7
        score += similarity_weight * (1 - distances[0][i])
        
        # weight of year
        year_weight = 0.2
        year = node_years[cluster_indices[article_index]]
        if year is not None:
            max_year = torch.max(node_years)
            score += year_weight * (year / max_year)
        
        relevant_scores.append((article_data, score))
    
    # arrange the articles in descending order of score
    relevant_scores.sort(key=lambda x: x[1], reverse=True)
    
    return relevant_scores

def precompute():
    origin_features = 64
    hidden_channels = 128
    node_features = 40
    num_clusters = 40

    # Load data
    adj_t, x, node_years, edge_index, index_to_data, index_to_id, vectorizer, unique_ids, id_to_index = data_preprocess.get_data(features=origin_features)
    adj_t = adj_t.to(device)
    x = x.to(device)

    # Create SAGE model
    model = SAGE(origin_features, hidden_channels, node_features, 3, 0.5).to(device)
    node_embeddings = model(x, adj_t)

    # Create mapping model
    mapping_model = train_mapping_model(x, node_embeddings)

    # transfer node embeddings to numpy array
    node_embeddings = node_embeddings.cpu().detach().numpy()

    # apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(node_embeddings)

    # get cluster labels and cluster centers
    labels = kmeans.labels_
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)

    print(f"Number of clusters: {num_clusters}")
    print(f"Cluster centers shape: {cluster_centers.shape}")
    for i in range(num_clusters):
        cluster_size = np.sum(labels == i)
        print(f"Cluster {i} size: {cluster_size}")

    with open('precomputed.pkl', 'wb') as f:
        pickle.dump((model, node_embeddings, mapping_model, kmeans, labels, cluster_centers, vectorizer, index_to_data, index_to_id, origin_features, unique_ids, node_years, id_to_index), f)

def main():
    try:
        with open('precomputed.pkl', 'rb') as f:
            model, node_embeddings, mapping_model, kmeans, labels, cluster_centers, vectorizer, index_to_data, index_to_id, origin_features, unique_ids, node_years, id_to_index = pickle.load(f)
    except FileNotFoundError:
        precompute()
        with open('precomputed.pkl', 'rb') as f:
            model, node_embeddings, mapping_model, kmeans, labels, cluster_centers, vectorizer, index_to_data, index_to_id, origin_features, unique_ids, node_years, id_to_index = pickle.load(f)

    title = "The assembly, regulation and function of the mitochondrial respiratory chain"
    abstract = "The mitochondrial oxidative phosphorylation system is central to cellular metabolism. It comprises five enzymatic complexes and two mobile electron carriers that work in a mitochondrial respiratory chain. By coupling the oxidation of reducing equivalents coming into mitochondria to the generation and subsequent dissipation of a proton gradient across the inner mitochondrial membrane, this electron transport chain drives the production of ATP, which is then used as a primary energy carrier in virtually all cellular processes. Minimal perturbations of the respiratory chain activity are linked to diseases; therefore, it is necessary to understand how these complexes are assembled and regulated and how they function. In this Review, we outline the latest assembly models for each individual complex, and we also highlight the recent discoveries indicating that the formation of larger assemblies, known as respiratory"
    keywords = ["", "", "", "", ""]

    new_article_embedding = embed_new_article(title, abstract, keywords, mapping_model, origin_features, vectorizer)

    nearest_cluster_index = find_nearest_cluster(torch.tensor(new_article_embedding), cluster_centers)
    print(f"The new article belongs to cluster {nearest_cluster_index}")

    # the reference set of the cluster
    reference_set = get_reference_set(nearest_cluster_index, labels, index_to_data, unique_ids)

    # rank the reference and relevant articles
    top_references = rank_references(reference_set, index_to_data, id_to_index, labels, node_years, new_article_embedding, node_embeddings, nearest_cluster_index)

    top_relevant = rank_relevant(index_to_data, id_to_index, labels, node_years, new_article_embedding, node_embeddings, nearest_cluster_index, index_to_id)

    # print the top 5 relevant and reference articles
    print("Top 5 relevant articles:")
    for i, (article_data, score) in enumerate(top_relevant[:5], start=1):
        print(f"{i}. Title: {article_data['title']}")
        print(f"   Journal: {article_data['journal']}")
        print(f"   doi: {article_data['doi']}")
        print(f"   pubDate: {article_data['pubDate']}")
        print(f"   Score: {score[0].item():.4f}")
        print()

    print("Top 5 reference articles:")
    for i, (article_data, score) in enumerate(top_references[:5], start=1):
        print(f"{i}. Title: {article_data['title']}")
        print(f"   Journal: {article_data['journal']}")
        print(f"   doi: {article_data['doi']}")
        print(f"   pubDate: {article_data['pubDate']}")
        print(f"   Score: {score[0].item():.4f}")
        print()

if __name__ == "__main__":
    main()