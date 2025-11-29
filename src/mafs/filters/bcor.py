import os
import time
import numpy as np
import pandas as pd
import torch
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class BCORFilter:
    def __init__(self, top_k=100, script_path=None, 
                 bcor_weight="chisquare", bcor_method="standard"):
        self.top_k = top_k
        self.bcor_weight = bcor_weight
        self.bcor_method = bcor_method
        
        if script_path is None:
            self.script_path = str(Path(__file__).parent / "bcor_script.R")
        else:
            self.script_path = script_path
        
        self.scores_ = None
        self.selected_indices_ = None
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"R script not found: {self.script_path}")
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            temp_data = pd.DataFrame(X)
            temp_data['label'] = y
            temp_data.to_csv(input_file, index=False)
            
            try:
                subprocess.run([
                    'Rscript', self.script_path,
                    input_file, output_file,
                    self.bcor_method, self.bcor_weight
                ], check=True, capture_output=True, timeout=3600)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"R script failed: {e.stderr.decode()}")
            except FileNotFoundError:
                raise RuntimeError("Rscript not found. Please install R.")
            
            if not os.path.exists(output_file):
                raise RuntimeError("R script did not produce output")
            
            result = pd.read_csv(output_file)
            self.scores_ = result[self.bcor_weight].values
        
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
                          bcor_method="standard", bcor_weight="chisquare",
                          script_path=None, data_type=None, y_type=None):
    start_time = time.time()
    
    os.makedirs(weights_path, exist_ok=True)
    temp_dir = os.path.join(weights_path, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    output_file = os.path.join(
        weights_path,
        f'bcor_weights_{data_type}_{y_type}_{dataset_name}_seed{seed}_{bcor_method}_{bcor_weight}.csv'
    )
    temp_filename = f"temp_data_{data_type}_{y_type}_{dataset_name}_seed{seed}.csv"
    input_file_path = os.path.join(temp_dir, temp_filename)
    
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    
    print(f"Computing BCOR scores using Ball correlation")
    
    try:
        temp_data = pd.DataFrame(data)
        temp_data['label'] = label
        temp_data.to_csv(input_file_path, index=False)
        
        r_command = [
            'Rscript', script_path, input_file_path, output_file,
            bcor_method, bcor_weight, str(seed)
        ]
        
        result = subprocess.run(r_command, check=True, capture_output=True, text=True)
        
        end_time = time.time()
        bcor_time = end_time - start_time
        
        selected_features = pd.read_csv(output_file)
        bcor_scores = selected_features[bcor_weight].values
        
        weight_df = pd.DataFrame({bcor_weight: bcor_scores})
        weight_df.to_csv(output_file, index=False)
        
        print(f"BCOR computation completed in {bcor_time:.2f}s")
        print(f"Score stats - Mean: {bcor_scores.mean():.6f}, "
              f"Max: {np.max(bcor_scores):.6f}, "
              f"Min: {np.min(bcor_scores):.6f}")
        
        return output_file, bcor_time
        
    finally:
        if os.path.exists(input_file_path):
            try:
                os.remove(input_file_path)
            except Exception:
                pass
