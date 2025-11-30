import os
import time
import numpy as np
import pandas as pd
from scipy import stats


class SISFilter:
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.scores_ = None
        self.selected_indices_ = None
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_features = X.shape[1]
        self.scores_ = np.zeros(n_features)
        
        for i in range(n_features):
            try:
                r, _ = stats.pearsonr(X[:, i], y)
                self.scores_[i] = 0 if np.isnan(r) else abs(r)
            except Exception:
                self.scores_[i] = 0
        
        self.selected_indices_ = np.argsort(self.scores_)[-self.top_k:]
        return self
    
    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices=True):
        if indices:
            return self.selected_indices_
        mask = np.zeros(len(self.scores_), dtype=bool)
        mask[self.selected_indices_] = True
        return mask
    
    def get_scores(self):
        return self.scores_


def calculate_sis_weights(data, label, weights_path, dataset_name, seed, 
                         data_type, y_type):
    start_time = time.time()
    
    # Convert to numpy if needed - handle both torch tensors and arrays
    if hasattr(data, 'cpu'):  # Check if it's a torch tensor
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)
    
    if hasattr(label, 'cpu'):  # Check if it's a torch tensor
        label = label.cpu().numpy()
    else:
        label = np.asarray(label)
    
    print("Computing SIS scores using Pearson correlation")
    
    os.makedirs(weights_path, exist_ok=True)
    output_file = os.path.join(
        weights_path, 
        f'sis_weights_{data_type}_{y_type}_{dataset_name}_seed{seed}.csv'
    )
    
    n_features = data.shape[1]
    sis_scores = np.zeros(n_features)
    
    for i in range(n_features):
        try:
            r, _ = stats.pearsonr(data[:, i], label)
            sis_scores[i] = 0 if np.isnan(r) else abs(r)
        except Exception:
            sis_scores[i] = 0
    
    end_time = time.time()
    sis_time = end_time - start_time
    
    weight_df = pd.DataFrame({"sis_scores": sis_scores})
    weight_df.to_csv(output_file, index=False)
    
    print(f"SIS computation completed in {sis_time:.2f}s")
    print(f"Score stats - Max: {np.max(sis_scores):.6f}, "
          f"Min: {np.min(sis_scores):.6f}, "
          f"Mean: {np.mean(sis_scores):.6f}")
    
    return output_file, sis_time
