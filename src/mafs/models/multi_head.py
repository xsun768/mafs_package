import numpy as np
import torch
from .model import SingleHeadSelector, train_single_head

class MultiHeadSelector:
    def __init__(self, input_size, n_classes, weight_files, 
                 hidden_scale=4, dropout_rate=0.2, 
                 y_type='categorical', device='cpu'):
        self.input_size = input_size
        self.n_classes = n_classes
        self.weight_files = weight_files
        self.hidden_scale = hidden_scale
        self.dropout_rate = dropout_rate
        self.y_type = y_type
        self.device = device
        
        self.n_heads = len(weight_files)
        self.models = []
        self.feature_weights = []
    
    def fit(self, train_loader, val_loader):
        all_results = []
        
        for head_idx, weight_file in enumerate(self.weight_files):
            model = SingleHeadSelector(
                input_size=self.input_size,
                n_classes=self.n_classes,
                weight_file=weight_file,
                hidden_scale=self.hidden_scale,
                dropout_rate=self.dropout_rate,
                y_type=self.y_type
            ).to(self.device)
            
            trained_model, final_weights = train_single_head(
                model, train_loader, val_loader, self.device
            )
            
            self.models.append(trained_model)
            self.feature_weights.append(final_weights)
            
            top_indices = np.argsort(final_weights.flatten())[-100:][::-1]
            
            all_results.append({
                'head_idx': head_idx,
                'model': trained_model,
                'weights': final_weights,
                'top_features': top_indices
            })
        
        return all_results
    
    def predict(self, X, head_idx=None):
        if head_idx is not None:
            model = self.models[head_idx]
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                output = model(X_tensor)
            return output.cpu().numpy()
        else:
            predictions = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    output = model(X_tensor)
                predictions.append(output.cpu().numpy())
            return np.mean(predictions, axis=0)


def train_multi_head(weight_files, train_loader, val_loader, 
                     input_size, n_classes, device='cpu'):
    multi_head = MultiHeadSelector(
        input_size=input_size,
        n_classes=n_classes,
        weight_files=weight_files,
        hidden_scale=200,
        dropout_rate=0.4,
        y_type='numerical',
        device=device
    )
    
    results = multi_head.fit(train_loader, val_loader)
    
    return multi_head, results
