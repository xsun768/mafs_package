import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class FeatureSelectionMLP(nn.Module):
    def __init__(self, input_size, weight_file=None):

        super().__init__()
        
        if weight_file is not None:
            # Load weights from file
            self.weights = self._load_weights(weight_file)
            self.normalized_weights = self._z_score_normalize(self.weights)
            self.selection_rate = nn.Parameter(self.normalized_weights.clone())
            print(f"Loaded weights from: {weight_file}")
        else:
            # Initialize with ones
            self.selection_rate = nn.Parameter(torch.ones([1, input_size]))
            print(f"Initialized with ones (no weight file)")
    
    def _load_weights(self, weight_file):
        """Load weights from CSV file"""
        weights_df = pd.read_csv(weight_file)
        scores = weights_df.iloc[:, 0].values
        return torch.tensor(scores, dtype=torch.float32).reshape(1, -1)
    
    def _z_score_normalize(self, weights):
        """Z-score normalization"""
        mean = torch.mean(weights)
        std = torch.std(weights, unbiased=False)
        
        if std < 1e-8:
            return weights
        
        return (weights - mean) / std
    
    def get_selection_rate(self):
        return torch.sigmoid(self.selection_rate)
    
    def forward(self, x):
        return x * torch.sigmoid(self.selection_rate)


class Classifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_scale=200, dropout_rate=0.4, y_type='categorical'):
  
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
    """Single-head feature selector with EARFS"""
    def __init__(self, input_size, n_classes, weight_file=None,hidden_scale=200, dropout_rate=0.4, y_type='categorical'):
        super().__init__()
        
        self.feature_selection = FeatureSelectionMLP(
            input_size=input_size,
            weight_file=weight_file
        )
        
        self.classifier = Classifier(
            input_size, 
            n_classes, 
            hidden_scale=hidden_scale, 
            dropout_rate=dropout_rate, 
            y_type=y_type
        )
        
    def forward(self, x):
        return self.classifier(self.feature_selection(x))
    
    def get_feature_weights(self):
        return self.feature_selection.get_selection_rate()


def train_single_head(model, train_loader, val_loader, device, 
                     y_type='categorical', reg_lambda=1e-2,epochs=100, lr=1e-5, weight_decay=1e-5):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, epochs, 1, eta_min=1e-3
    )
    
    if y_type == 'categorical':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    train_size = sum([x.size(0) for x, _ in train_loader])
    val_size = sum([x.size(0) for x, _ in val_loader])
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if y_type != 'categorical':
                targets = targets.float().view(-1, 1)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            
        
            selection_rate = model.get_feature_weights()
            reg_loss = reg_lambda / torch.sum((selection_rate - 0.5) ** 2)
            
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * inputs.size(0)
            
    
            if y_type == 'categorical':
                predicts = output.argmax(dim=1)
                train_acc += torch.eq(predicts, targets).sum().float().item()
        
        train_loss /= train_size
        scheduler.step()
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0.0
            val_acc = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    if y_type != 'categorical':
                        targets = targets.float().view(-1, 1)
                    
                    output = model(inputs)
                    loss = criterion(output, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    if y_type == 'categorical':
                        predicts = output.argmax(dim=1)
                        val_acc += torch.eq(predicts, targets).sum().float().item()
            
            val_loss /= val_size
    
    final_weights = model.get_feature_weights().detach().cpu().numpy()
    
    return model, final_weights
