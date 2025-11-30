import numpy as np
import torch

# Import from model.py
try:
    from model import SingleHeadSelector, train_single_head
except ImportError:
    from .model import SingleHeadSelector, train_single_head

# Import evaluator
try:
    from evaluator import FeatureSelectionEvaluator
except ImportError:
    FeatureSelectionEvaluator = None
    print("Warning: evaluator module not found, evaluation features disabled")


class MultiHeadSelector:
    def __init__(self, input_size, n_classes, weight_files, hidden_scale=200, dropout_rate=0.4, 
                 y_type='categorical', device='cpu',data_file_path=None):
        
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
        
        # Initialize evaluator if data file provided
        self.evaluator = None
        if data_file_path and FeatureSelectionEvaluator:
            self.evaluator = FeatureSelectionEvaluator(data_file_path)
    
    def fit(self, train_loader, val_loader):
        """Train all heads"""
        all_results = []
        
        for head_idx, weight_file in enumerate(self.weight_files):
            print(f"\nTraining head {head_idx+1}/{self.n_heads}...")
            
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
    
    def evaluate(self, top_k_list=[100, 200, 300, 500], method='mean', print_results=True):
        if self.evaluator is None:
            print("No evaluator available. Provide data_file_path when creating MultiHeadSelector.")
            return None
        
        # Get selected features
        selected_features = self.get_selected_features(method=method)
        
        # Evaluate
        results = self.evaluator.evaluate(selected_features, top_k_list)
        
        # Print if requested
        if print_results and results:
            self.evaluator.print_results(results, title=f"EVALUATION RESULTS (method={method})")
        
        return results
    
    def predict(self, X, head_idx=None):
        """Make predictions"""
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
                     input_size, n_classes, device='cpu',
                     data_file_path=None, evaluate=True):
    multi_head = MultiHeadSelector(
        input_size=input_size,
        n_classes=n_classes,
        weight_files=weight_files,
        hidden_scale=4,
        dropout_rate=0.2,
        y_type='numerical',
        device=device,
        data_file_path=data_file_path
    )
    
    # Train
    results = multi_head.fit(train_loader, val_loader)
    
    # Evaluate if requested
    eval_results = None
    if evaluate and data_file_path:
        print("\n" + "="*80)
        print("EVALUATING FEATURE SELECTION")
        print("="*80)
        eval_results = multi_head.evaluate()
    
    return multi_head, results, eval_results
