
from model_definition import Classifier
import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F



x_dict = torch.load('x_dict.pth')
len(x_dict['perturbation'])

node_features = np.zeros((2, 128), dtype=int)
node_features

therapeutic_node_embedding = x_dict["perturbation"][-1]
therapeutic_node_embedding = torch.unsqueeze(therapeutic_node_embedding, 0)

gene_embeddings =  x_dict["gene"]

edge_index = np.zeros((2, 18315), dtype=int)
#edge_index[0, -1] = 1.0
edge_index[1,:] = np.arange(0, 18315)
#edge_index[1,18315:] = np.arange(0, 3)
edge_label_index = torch.tensor(edge_index)


classifier = Classifier()
classifier.load_state_dict(torch.load('classifier_trained_model.pth'))


classifier.eval()
with torch.no_grad():
    dot_products = classifier(therapeutic_node_embedding, gene_embeddings, edge_label_index)

dot_products = dot_products.to('cpu')

edge_labels = torch.tensor(np.ones(18315))


edge_losses = []
for i in range(len(edge_labels)):
    edge_label = edge_labels[i]
    dot_product = dot_products[i]
    edge_loss = F.binary_cross_entropy_with_logits(dot_product, edge_label)
    edge_losses.append(float(edge_loss))
edge_losses = torch.tensor(edge_losses)


predicted_edge_scores = 1 - edge_losses



k = 20
# Use torch.topk to get the indices and values of the k smallest elements
top_edge_scores, top_indices = torch.topk(predicted_edge_scores, k)
top_edge_scores = top_edge_scores.tolist()
top_indices = top_indices.tolist()
    

# gene_mapping_dictionary from json file, consider datatypes in the dictionary!

file_path = 'mapping_dicts/genes_dict.json'
with open(file_path, 'r') as json_file:
    genes_dict_json = json.load(json_file)

#convert back strings into integers as they got mutated in json conversion! 
    genes_dict = {value: int(key) for key, value in genes_dict_json.items()}
    
gene_names = pd.read_table("gene_names.txt")
gene_names = gene_names.dropna()
gene_names["entrezId"] = gene_names["entrezId"].astype("int")
gene_names = gene_names.set_index("entrezId")



gene_ids = []
gene_symbols =[]
scores = []
for i in range(len(top_indices)):
    gene_id = genes_dict[top_indices[i]]
    gene_ids.append(gene_id)
    score = round(top_edge_scores[i], 3)
    scores.append(score)
    if gene_id in gene_names["geneSymbol"]:
        gene_symbol = gene_names.loc[gene_id, "geneSymbol"]
    else:
        gene_symbol = "missing"
    gene_symbols.append(gene_symbol)



# Open the file in append mode
with open('conv_output.txt', 'a') as file:
    # Append output to the file
    for i in range(len(gene_ids)):
        line = f"{str(gene_ids[i])},{str(gene_symbols[i])},{str(scores[i])}"
        line += "\n"
        file.write(line)

