import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import tqdm




# Load node feature matrices and convert them into torch tensors
perturbations = np.load('node_features/perts_pca_node_features.npy')
perturbations = torch.from_numpy(perturbations)

genes = np.load('node_features/gene_node_feature_matrix.npy')
genes = torch.from_numpy(genes)

#mirna = np.load('node_features/mirna_node_feature_matrix.npy')
#mirna = torch.from_numpy(mirna)

tissue = np.load('node_features/tissue_node_feature_matrix.npy')
tissue = torch.from_numpy(tissue)

pw = np.load('node_features/pw_node_feature_matrix.npy')
pw = torch.from_numpy(pw)

mf = np.load('node_features/mf_node_feature_matrix.npy')
mf = torch.from_numpy(mf)

bp = np.load('node_features/bp_node_feature_matrix.npy')
bp = torch.from_numpy(bp)




# Load edge indeces

perturbation_trtxpr_gene = torch.from_numpy(np.load('edge_index/xpr_uniq_genes_edge_index.npy'))


gene_activation_gene = torch.from_numpy(np.load('edge_index/edge_index_activation.npy'))
gene_inhibition_gene = torch.from_numpy(np.load('edge_index/edge_index_inhibition.npy'))
gene_catalysis_gene = torch.from_numpy(np.load('edge_index/edge_index_catalysis.npy'))
gene_binding_gene = torch.from_numpy(np.load('edge_index/edge_index_binding.npy'))
gene_reaction_gene = torch.from_numpy(np.load('edge_index/edge_index_reaction.npy'))
gene_expression_gene = torch.from_numpy(np.load('edge_index/edge_index_expression.npy'))
gene_ptmod_gene = torch.from_numpy(np.load('edge_index/edge_index_ptmod.npy'))
gene_other_gene = torch.from_numpy(np.load('edge_index/edge_index_other.npy'))

#gene_influenced_by_mirna = torch.from_numpy(np.load('edge_index/mirna_edge_index.npy'))


gene_expressed_in_tissue = torch.from_numpy(np.load('edge_index/gene_expressed_tissue_edge_index.npy'))
gene_upregulated_in_tissue = torch.from_numpy(np.load('edge_index/gene_upregulated_tissue_edge_index.npy'))
gene_downregulated_in_tissue = torch.from_numpy(np.load('edge_index/gene_downregulated_tissue_edge_index.npy'))


gene_in_pw = torch.from_numpy(np.load('edge_index/pw_gene_edge_index.npy'))
gene_in_mf = torch.from_numpy(np.load('edge_index/mf_gene_edge_index.npy'))
gene_in_bp = torch.from_numpy(np.load('edge_index/bp_gene_edge_index.npy'))







data = HeteroData() 

data['perturbation'].x = perturbations # [num_perturbation, num_features_perturbation]
data["perturbation"].x = data["perturbation"].x.float()



data["perturbation"].node_id = torch.arange(len(data['perturbation'].x))
data["gene"].node_id = torch.arange(len(genes))
data["tissue"].node_id = torch.arange(len(tissue))
data["pw"].node_id = torch.arange(len(pw))
data["mf"].node_id = torch.arange(len(mf))
data["bp"].node_id = torch.arange(len(bp))


data['perturbation', 'trtxpr', 'gene'].edge_index = perturbation_trtxpr_gene # [2, num_edges]


data['gene', 'activates', 'gene'].edge_index = gene_activation_gene # [2, num_edges]
data['gene', 'inhibits', 'gene'].edge_index = gene_inhibition_gene # [2, num_edges]
data['gene', 'catalyzes', 'gene'].edge_index = gene_catalysis_gene # [2, num_edges]
data['gene', 'binds_to', 'gene'].edge_index = gene_binding_gene # [2, num_edges]
data['gene', 'reacts_with', 'gene'].edge_index = gene_reaction_gene # [2, num_edges]
data['gene', 'expressed_with', 'gene'].edge_index = gene_expression_gene # [2, num_edges]
data['gene', 'ptmod', 'gene'].edge_index = gene_ptmod_gene # [2, num_edges]
data['gene', 'other', 'gene'].edge_index = gene_other_gene # [2, num_edges]

#data['gene', 'influenced_by', 'mirna'].edge_index = gene_influenced_by_mirna # [2, num_edges]

data['gene', 'expressed_in', 'tissue'].edge_index = gene_expressed_in_tissue # [2, num_edges]
data['gene', 'upregulated_in', 'tissue'].edge_index = gene_upregulated_in_tissue # [2, num_edges]
data['gene', 'downregulated_in', 'tissue'].edge_index = gene_downregulated_in_tissue # [2, num_edges]

data['gene', 'involved_in', 'pw'].edge_index = gene_in_pw # [2, num_edges]
data['gene', 'associated_with', 'mf'].edge_index = gene_in_mf # [2, num_edges]
data['gene', 'participates_in', 'bp'].edge_index = gene_in_bp # [2, num_edges]




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = T.ToUndirected()
data = transform(data)
data = data.to(device)





# split the set of edges into training (80%), validation (10%), and testing edges (10%).
# we use 70% of the training edges for message passing and 30% of the edges for supervision.
# We further want to generate negative edges for evaluation with a ratio of 2:1.

transform_2 = T.RandomLinkSplit(
    num_val = int(0.1 * 5527),  
    num_test = int(0.1 * 5527), 
    disjoint_train_ratio = 0.7,  
    neg_sampling_ratio = 2.0,  
    add_negative_train_samples = False,
    edge_types = ("perturbation", "trtxpr", "gene"),
    rev_edge_types = ("gene", "rev_trtxpr", "perturbation")
)

#transform = T.RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform_2(data)



from torch_geometric.loader import LinkNeighborLoader

# Define seed edges:
edge_label_index = train_data["perturbation", "trtxpr", "gene"].edge_label_index
edge_label = train_data["perturbation", "trtxpr", "gene"].edge_label


train_loader = LinkNeighborLoader(
    train_data,  # TODO
    num_neighbors=[20,10],  # TODO
    neg_sampling_ratio=2,  # TODO
    edge_label_index=(("perturbation", "trtxpr", "gene"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)

# Inspect a sample:
sampled_data = next(iter(train_loader))

#print("Sampled mini-batch:")
#print("===================")
#print(sampled_data)




from torch_geometric.nn import GraphConv, to_hetero
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
       
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
  
        return x
     

        
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_perturbation: Tensor, x_gene: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_perturbation = x_perturbation[edge_label_index[0]]
        edge_feat_gene = x_gene[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_perturbation * edge_feat_gene).sum(dim=-1)

    
# model assembly 
class DeepTarg(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()

        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for perturbations and genes:
        self.data = data
        self.perturbation_lin = torch.nn.Linear(512, hidden_channels)
        self.gene_emb = torch.nn.Embedding(self.data["gene"].num_nodes, hidden_channels)
        self.tissue_emb = torch.nn.Embedding(self.data["tissue"].num_nodes, hidden_channels)
        self.pw_emb = torch.nn.Embedding(self.data["pw"].num_nodes, hidden_channels)
        self.mf_emb = torch.nn.Embedding(self.data["mf"].num_nodes, hidden_channels)
        self.bp_emb = torch.nn.Embedding(self.data["bp"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=self.data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {"perturbation": self.perturbation_lin(self.data["perturbation"].x.float()), 
                  "gene": self.gene_emb(self.data["gene"].node_id),
                  "tissue": self.tissue_emb(self.data["tissue"].node_id), 
                  "pw": self.pw_emb(self.data["pw"].node_id), 
                  "mf": self.mf_emb(self.data["mf"].node_id), 
                  "bp": self.bp_emb(self.data["bp"].node_id)}
                    
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        classifier_state_dict = self.classifier.state_dict()
        gnn_state_dict = self.gnn.state_dict()
        
        
        pred = self.classifier(
            x_dict["perturbation"],
            x_dict["gene"],
            data["perturbation", "trtxpr", "gene"].edge_label_index,
        )

        return pred, x_dict, gnn_state_dict, classifier_state_dict


model = DeepTarg(data, hidden_channels=512)
#print('model', model)







model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


gnn_state_dict = {}
classifier_state_dict = {}
x_dict = {}
for epoch in range(1, 25):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()

        # Move `sampled_data` to the respective `device`
        sampled_data = sampled_data.to(device)

        # Run `forward` pass of the model
        pred = model(sampled_data)[0]

        
        # Ground truth can be obtained from the sampled_data
        ground_truth = sampled_data["perturbation", "trtxpr", "gene"].edge_label.view(-1, 1).float()[:,0]
   
        # Apply binary cross entropy
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        
        x_dict = model(sampled_data)[1]
        gnn_state_dict = model(sampled_data)[2]
        classifier_state_dict = model(sampled_data)[3]
    #print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

#Save the trained mode
dummy_state_dict = model.state_dict()
torch.save(dummy_state_dict, 'DeepTarg_trained_model.pth')

# save x_dict as json file
torch.save(x_dict, 'x_dict.pth')
torch.save(gnn_state_dict, 'gnn_trained_model.pth')
torch.save(classifier_state_dict, 'classifier_trained_model.pth')
    
    
ground_truth = sampled_data["perturbation", "trtxpr", "gene"].edge_label.view(-1, 1)


# Define the validation seed edges:
edge_label_index = val_data["perturbation", "trtxpr", "gene"].edge_label_index
edge_label = val_data["perturbation", "trtxpr", "gene"].edge_label

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("perturbation", "trtxpr", "gene"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)

sampled_data = next(iter(val_loader))




# model evaluation
from sklearn.metrics import roc_auc_score


preds = []
ground_truths = []

for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        
        sampled_data = sampled_data.to(device)
        # Forward pass to get predictions
        pred = model(sampled_data)[0]
     
        
        
        # Collect predictions and ground-truths
        preds.append(pred.flatten())
        ground_truths.append(sampled_data["perturbation", "trtxpr", "gene"].edge_label.view(-1))
        
       
        
# Concatenate predictions and ground-truths
preds = torch.cat(preds, dim=0).cpu().numpy()
ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()

np.save("predictions.npy", preds)
np.save("ground_truths.npy", ground_truths)

# Calculate ROC AUC score
auc = roc_auc_score(ground_truths, preds)

#print()
#print(f"Validation AUC: {auc:.4f}")
 





