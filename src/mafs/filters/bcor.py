import os
import time
import numpy as np
import pandas as pd
import subprocess
import tempfile


class BCORFilter:
    """
    Ball Correlation based feature selection using R's Ball package
    Uses bcorsis function (sequential processing)
    """
    def __init__(self, top_k=100, method="standard", weight="chisquare"):
        self.top_k = top_k
        self.method = method  # "standard" or other Ball package methods
        self.weight = weight  # "constant", "probability", or "chisquare"
        self.scores_ = None
        self.selected_indices_ = None
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Find bcorsis.R script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        r_script = os.path.join(script_dir, 'bcorsis.R')
        
        if not os.path.exists(r_script):
            raise FileNotFoundError(f"bcorsis.R not found at {r_script}")
        
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_in:
            with tempfile.NamedTemporaryFile(mode='r', suffix='.csv', delete=False) as f_out:
                
                data = np.column_stack([X, y])
                np.savetxt(f_in.name, data, delimiter=',')
                
                result = subprocess.run(
                    ['Rscript', r_script, f_in.name, f_out.name, 
                     self.method, self.weight],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode != 0:
                    print(f"R script stderr: {result.stderr}")
                    raise RuntimeError(f"BCOR computation failed: {result.stderr}")
                
                scores_df = pd.read_csv(f_out.name)
                
                self.scores_ = scores_df[self.weight].values
                
                os.unlink(f_in.name)
                os.unlink(f_out.name)
        
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


def calculate_bcor_weights(data, label, weights_path, dataset_name, seed, 
                          data_type, y_type, method="standard", weight="chisquare"):
    start_time = time.time()
    
    # Convert to numpy
    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)
    
    if hasattr(label, 'cpu'):
        label = label.cpu().numpy()
    else:
        label = np.asarray(label)
    
    print(f"Computing BCOR scores using Ball package (sequential)")
    print(f"  Method: {method}, Weight: {weight}")
    
    os.makedirs(weights_path, exist_ok=True)
    output_file = os.path.join(
        weights_path, 
        f'bcor_weights_{data_type}_{y_type}_{dataset_name}_seed{seed}.csv'
    )
    
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_script = os.path.join(script_dir, 'bcorsis.R')
    
    if not os.path.exists(r_script):
        raise FileNotFoundError(f"bcorsis.R not found at {r_script}")
    
    n_features = data.shape[1]
    print(f"  Computing BCOR for {n_features} features...")
    
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_in:
        
        combined_data = np.column_stack([data, label])
        np.savetxt(f_in.name, combined_data, delimiter=',')
        
    
        result = subprocess.run(
            ['Rscript', r_script, f_in.name, output_file, method, weight],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"R script stderr: {result.stderr}")
            raise RuntimeError(f"BCOR computation failed: {result.stderr}")
        
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        os.unlink(f_in.name)
    
    end_time = time.time()
    bcor_time = end_time - start_time
    
    scores_df = pd.read_csv(output_file)
    bcor_scores = scores_df[weight].values
    
    print(f"BCOR computation completed in {bcor_time:.2f}s")
    print(f"Score stats - Max: {np.max(bcor_scores):.6f}, "
          f"Min: {np.min(bcor_scores):.6f}, "
          f"Mean: {np.mean(bcor_scores):.6f}")
    
    return output_file, bcor_time
