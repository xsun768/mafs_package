import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class FeatureSelection(nn.Module):
    def __init__(self, weight_file, input_size):
        super().__init__()
        self.weights = self._load_weights(weight_file)
        self.normalized_weights = self._z_score_normalize(self.weights)
        self.selection_rate = nn.Parameter(self.normalized_weights.clone())
    
    def _load_weights(self, weight_file):
        weights_df = pd.read_csv(weight_file)
        scores = weights_df.iloc[:, 0].values
        return torch.tensor(scores, dtype=torch.float32).reshape(1, -1)
    
    def _z_score_normalize(self, weights):
        mean = torch.mean(weights)
        std = torch.std(weights, unbiased=False)
        
        if std < 1e-8:
            return weights
        
        return (weights - mean) / std
    
    def get_selection_rate(self):
        return torch.relu(self.selection_rate)
    
    def forward(self, x):
        return x * torch.relu(self.selection_rate)


class Classifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_scale=200, 
                 dropout_rate=0.4, y_type='categorical'):
        super().__init__()
        
        hidden_size = int(input_size / hidden_scale)
        
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_rate),
        ]
        
        if y_type == 'categorical':
            layers.append(nn.Linear(hidden_size, n_classes))
        else:
            layers.append(nn.Linear(hidden_size, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class SingleHeadSelector(nn.Module):
    def __init__(self, input_size, n_classes, weight_file, 
                 hidden_scale=4, dropout_rate=0.2, y_type='categorical'):
        super().__init__()
        self.feature_selection = FeatureSelection(weight_file, input_size)
        self.classifier = Classifier(
            input_size, n_classes, hidden_scale, 
            dropout_rate, y_type
        )
    
    def forward(self, x):
        return self.classifier(self.feature_selection(x))
    
    def get_feature_weights(self):
        return self.feature_selection.get_selection_rate()


def train_single_head(model, train_loader, val_loader, device,y_type='categorical', reg_lambda=1e-5,gamma=0.5):
    lr = 1e-5
    epochs = 100
    weight_decay = 1e-5
    patience = 10
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, epochs, 1, eta_min=1e-3
    )
    
    if y_type == 'categorical':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    initial_weights = model.get_feature_weights().detach().cpu().numpy()
    epsilon = 1e-5
    tau = 1.0 / (torch.abs(torch.tensor(initial_weights, device=device)) + epsilon) ** gamma
    tau = torch.clamp(tau, max=20.0).squeeze()
    
    train_size = sum([x.size(0) for x, _ in train_loader])
    val_size = sum([x.size(0) for x, _ in val_loader])
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if y_type != 'categorical':
                targets = targets.float().view(-1, 1)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            
            selection_rate = model.get_feature_weights().squeeze()
            reg_loss = reg_lambda * torch.sum(tau * torch.abs(selection_rate))
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * inputs.size(0)
        
        train_loss /= train_size
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if y_type != 'categorical':
                    targets = targets.float().view(-1, 1)
                
                output = model(inputs)
                loss = criterion(output, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= val_size
    
    final_weights = model.get_feature_weights().detach().cpu().numpy()
    
    return model, final_weights
