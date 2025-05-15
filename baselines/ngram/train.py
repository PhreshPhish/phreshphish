import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import classification_report

device = "cuda:1"

class SparseDataset(Dataset):
    def __init__(self, file_path, feature_count):
        self.data = []
        self.labels = []
        self.feature_count = feature_count
        print("loading data file", file_path)
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])
                indices = [int(p.split(":")[0]) for p in parts[1:]]
                self.data.append(indices)
                self.labels.append(1 if label == 1 else 0)
                if len(self.data) % 1000 == 0:
                    print(f"Read {len(self.data)} lines...")
                
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.zeros(self.feature_count)
        x[self.data[idx]] = 1.0
        return x, self.labels[idx]

class PhishingClassifier(nn.Module):
    ndim = 32
    def __init__(self, input_dim):
        super(PhishingClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def get_line_count(input_file):
    l = 0
    with open(input_file) as inf:
        for _ in inf: l += 1
    return l

def train_model(train_file, dict_file, model_path, batch_size, epochs):
    feature_count = get_line_count(dict_file)
    os.makedirs(model_path, exist_ok=True)
    
    # Load datasets
    train_dataset = SparseDataset(train_file, feature_count)    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = PhishingClassifier(feature_count).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):  # number of epochs can be adjusted
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            #print(features[0].shape)
            labels = labels.to(device)
            outputs = model(features).squeeze(dim=1)
            #print("output:", outputs.shape)
            #print("labels:", labels.shape)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch} loss {loss.item()}")

    # Save model
    model_save_path = os.path.join(model_path, 'model.pt')
    print("Saving model to", model_save_path)
    torch.save(model.state_dict(), model_save_path)

def test_model(test_file, dict_file, label_dict_file, model_path, report_file, batch_size):

    feature_count = get_line_count(dict_file)
    with open(label_dict_file) as inf:
        label_names = [r.strip() for r in inf.readlines()]
    # load model
    model = PhishingClassifier(feature_count).to(device)
    model_save_path = os.path.join(model_path, 'model.pt')
    model.load_state_dict(torch.load(model_save_path))
    # Evaluate
    model.eval()

    test_dataset = SparseDataset(test_file, feature_count)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features).squeeze()
            preds = (outputs >= 0.5).long().cpu()
            if (len(preds.shape)) > 0:
                all_preds.extend(preds.tolist())
            else:
                all_preds.append(preds)

            all_labels.extend(labels.cpu())

    # Generate report
    #print(all_preds)
    #print(all_labels)
    report = classification_report(all_labels, all_preds, target_names=label_names)
    print(report)
    with open(report_file, 'w') as f:
        f.write(report)
