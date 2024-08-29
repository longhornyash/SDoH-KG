import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

df = pd.read_csv('combined_graph_with_kg_and_primekg.csv')

all_nodes = pd.concat([df['x_name'], df['y_name']]).unique()
node_idx = {node: i for i, node in enumerate(all_nodes)}
edges = [[node_idx[x], node_idx[y]] for x, y in zip(df['x_name'], df['y_name'])]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

edge_labels = torch.ones(edge_index.size(1), dtype=torch.float)

num_negative_samples = edge_labels.size(0)
negative_samples = []
while len(negative_samples) < num_negative_samples:
    i = torch.randint(0, len(node_idx), (2,))
    if i[0] != i[1] and i.tolist() not in edges:
        negative_samples.append(i.tolist())
negative_samples = torch.tensor(negative_samples, dtype=torch.long).t().contiguous()

full_edge_index = torch.cat([edge_index, negative_samples], dim=1)
full_labels = torch.cat([edge_labels, torch.zeros(negative_samples.size(1))])

train_edges, test_edges, train_labels, test_labels = train_test_split(full_edge_index.t(), full_labels, test_size=0.2, random_state=42)

class GCN(nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(len(node_idx), hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)
        self.lin = nn.Linear(2 * hidden_channels, 1) 

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        return self.lin(edge_features).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

x = torch.eye(len(node_idx), dtype=torch.float).to(device)

train_edges = train_edges.t().to(device)
test_edges = test_edges.t().to(device)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(x, train_edges)
    loss = criterion(out, train_labels)
    loss.backward()
    optimizer.step()
    return loss

for epoch in range(50):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(x, test_edges)
        predictions = torch.sigmoid(out).round()
        accuracy = (predictions == test_labels).float().mean()
        precision = precision_score(test_labels.cpu(), predictions.cpu())
        recall = recall_score(test_labels.cpu(), predictions.cpu())
        auc = roc_auc_score(test_labels.cpu(), torch.sigmoid(out).cpu())
        fpr, tpr, _ = roc_curve(test_labels.cpu(), torch.sigmoid(out).cpu())
        conf_matrix = confusion_matrix(test_labels.cpu(), predictions.cpu())
        return accuracy, precision, recall, auc, fpr, tpr, conf_matrix

accuracy, precision, recall, auc, fpr, tpr, conf_matrix = evaluate()
print(f'Test Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'AUC: {auc:.2f}')


onehot_encoder = OneHotEncoder(sparse=False)
SDoH_encoded = onehot_encoder.fit_transform(df[['SDoH_factors']])
SDoH_feature_labels = ['SDoH_' + str(i) for i in range(SDoH_encoded.shape[1])]
num_nodes = len(node_idx)
num_features = num_nodes + SDoH_encoded.shape[1]
x = torch.zeros(num_nodes, num_features)
for i, node in enumerate(all_nodes):
    x[i, i] = 1  # Identity matrix part
    if node in df['x_name'].values:
        idx = df[df['x_name'] == node].index[0]
        x[i, num_nodes:] = torch.tensor(SDoH_encoded[idx])  


edges = [[node_idx[x], node_idx[y]] for x, y in zip(df['x_name'], df['y_name'])]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_weights = torch.rand(edge_index.size(1))  
edge_weights /= edge_weights.max() 


edge_labels = torch.ones(edge_index.size(1), dtype=torch.float)
num_negative_samples = edge_labels.size(0)
negative_samples = []
negative_weights = []
while len(negative_samples) < num_negative_samples:
    i = torch.randint(0, len(node_idx), (2,))
    if i[0] != i[1] and i.tolist() not in edges:
        negative_samples.append(i.tolist())
        negative_weights.append(torch.rand(1) * 0.1) 
negative_samples = torch.tensor(negative_samples, dtype=torch.long).t().contiguous()
negative_weights = torch.tensor(negative_weights, dtype=torch.float)


full_edge_index = torch.cat([edge_index, negative_samples], dim=1)
full_labels = torch.cat([edge_labels, torch.zeros(negative_samples.size(1))])
full_edge_weights = torch.cat([edge_weights, negative_weights])


train_data, test_data = train_test_split(
    torch.cat([full_edge_index.t(), full_edge_weights.unsqueeze(1), full_labels.unsqueeze(1)], dim=1),
    test_size=0.2, random_state=42)
train_edges, train_weights, train_labels = train_data[:, :2].long(), train_data[:, 2], train_data[:, 3]
test_edges, test_weights, test_labels = test_data[:, :2].long(), test_data[:, 2], test_data[:, 3]

class GCN_NoEdgeWeights(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(GCN_NoEdgeWeights, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)
        self.lin = nn.Linear(2 * hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        return torch.sigmoid(self.lin(edge_features)).squeeze()

class GCN_WithEdgeWeights(GCN_NoEdgeWeights):
    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.dropout(x)
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        return torch.sigmoid(self.lin(edge_features)).squeeze()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_no_weights = GCN_NoEdgeWeights(num_features, 64).to(device)
model_with_weights = GCN_WithEdgeWeights(num_features, 64).to(device)
optimizer_no_weights = torch.optim.Adam(model_no_weights.parameters(), lr=0.01)
optimizer_with_weights = torch.optim.Adam(model_with_weights.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

x = x.to(device)
train_edges = train_edges.t().to(device)
test_edges = test_edges.t().to(device)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)
train_weights = train_weights.to(device)
test_weights = test_weights.to(device)

def run_epoch(model, optimizer, criterion, edges, labels, weights=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()
    if weights is not None:
        outputs = model(x, edges, weights)
    else:
        outputs = model(x, edges)

    loss = criterion(outputs, labels)
    if is_train:
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = torch.sigmoid(outputs).round()
        accuracy = (predictions == labels).float().mean().item()
        precision = precision_score(labels.cpu(), predictions.cpu())
        recall = recall_score(labels.cpu(), predictions.cpu())
        roc_auc = roc_auc_score(labels.cpu(), torch.sigmoid(outputs).cpu())

    return loss.item(), accuracy, precision, recall, roc_auc, outputs, labels

num_epochs = 100
for epoch in range(num_epochs):
    loss_no_weights, acc_no_weights, _, _, _, _, _ = run_epoch(model_no_weights, optimizer_no_weights, criterion, train_edges, train_labels, is_train=True)
    loss_with_weights, acc_with_weights, _, _, _, outputs_w, labels_w = run_epoch(model_with_weights, optimizer_with_weights, criterion, train_edges, train_labels, train_weights, is_train=True)

    if epoch % 10 == 0:
        _, test_acc_no_weights, precision_nw, recall_nw, roc_auc_nw, outputs_nw, labels_nw = run_epoch(model_no_weights, optimizer_no_weights, criterion, test_edges, test_labels, is_train=False)
        _, test_acc_with_weights, precision_w, recall_w, roc_auc_w, outputs_w, labels_w = run_epoch(model_with_weights, optimizer_with_weights, criterion, test_edges, test_labels, test_weights, is_train=False)
        print(f'Epoch {epoch + 1}:')
        print(f'  Train Loss (No Weights): {loss_no_weights:.4f}, Accuracy: {acc_no_weights:.2f}')
        print(f'  Train Loss (With Weights): {loss_with_weights:.4f}, Accuracy: {acc_with_weights:.2f}')
        print(f'  Test Metrics (No Weights) - Accuracy: {test_acc_no_weights:.2f}, Precision: {precision_nw:.2f}, Recall: {recall_nw:.2f}, ROC-AUC: {roc_auc_nw:.2f}')
        print(f'  Test Metrics (With Weights) - Accuracy: {test_acc_with_weights:.2f}, Precision: {precision_w:.2f}, Recall: {recall_w:.2f}, ROC-AUC: {roc_auc_w:.2f}')