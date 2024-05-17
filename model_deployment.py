import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import tqdm
import json


AD_therapeutic_node_feature = np.load('node_features/pca_synthetic_node_features.npy')
print(AD_therapeutic_node_feature)

placeholder_node_feature = np.zeros(512).astype(int)
node_features = np.vstack((AD_therapeutic_node_feature, placeholder_node_feature))


#Sythetic nodes

nodes = ['AD_therapeutic', 'placeholder_node']

# Step 2: Prepare HeteroData Object
data_pred = HeteroData()

data_pred["perturbation"].x = torch.tensor(node_features)
data_pred["perturbation"].node_id = torch.arange(len(nodes))
data_pred["gene"].node_id = torch.arange(len(genes))


# Create a tensor with dimensions [2, 2440]
edge_index = np.zeros((2, 18315 + 3), dtype=int)

# Set the first half of the first row to ones
edge_index[0, 18315:] = 1.0
edge_index[1, 0:18315] = np.arange(0, 18315)
edge_index[1,18315:] = np.arange(0, 3)


transform = T.ToUndirected()
data = transform(data_pred)


data["perturbation", "trtxpr", "gene"].edge_label = torch.ones(18315 + 3)



data = torch.load('hetero_data.pt')


from xpr_uniq_DeepTarg import DeepTarg
model = DeepTarg(data, hidden_channels=128)
#Load the pre-trained model
model.load_state_dict(torch.load('DeepTarg_trained_model.pth'))

d_pred = HeteroData()

d_pred["perturbation"].node_id = torch.arange(2)
d_pred['perturbation'].x = torch.tensor(node_features) # [num_perturbation, num_features_perturbation]
d_pred['perturbation', 'trtxpr', 'gene'].edge_index = torch.tensor(edge_index) # [2, num_edges]
d_pred['perturbation', 'trtxpr', 'gene'].edge_label = torch.ones(18316)

transform = T.ToUndirected()
d_pred = transform(d_pred)
d_pred


data["perturbation"].node_id = d_pred["perturbation"].node_id
data['perturbation'].x = d_pred['perturbation'].x # [num_perturbation, num_features_perturbation]
data['perturbation', 'trtxpr', 'gene'].edge_index = d_pred['perturbation', 'trtxpr', 'gene'].edge_index # [2, num_edges]
data['gene', 'rev_trtxpr', 'perturbation'].edge_index = d_pred['gene', 'rev_trtxpr', 'perturbation'].edge_index 
data['perturbation', 'trtxpr', 'gene'].edge_label = torch.ones(18318)
data['perturbation', 'trtxpr', 'gene'].edge_label_index = torch.tensor(edge_index)



# Forward pass with the trained model
with torch.no_grad():
    model.eval()
    predictions = model(data)
    pred = predictions #[:610]
#threshold = 0.9
#filtered_links = predictions > threshold

# Print or use the filtered links as needed
#print(predictions[filtered_links])
# You can use predictions for further analysis
print(len(pred))
#print(len(predictions[filtered_links]))


#threshold = -70
#filtered_links = pred > threshold

# Print or use the filtered links as needed
#print(pred[filtered_links])


probabilistic_output = torch.sigmoid(pred)

k = 10
# Use torch.topk to get the indices and values of the k largest elements
top_values, top_indices = torch.topk(probabilistic_output, k)
print("top_value_indeces:", top_values, top_indices)

print("pred:", pred)