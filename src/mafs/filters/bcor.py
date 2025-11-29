"""Ball Correlation filter."""

import os
import numpy as np
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class BCORFilter:
    """
    Ball Correlation filter (requires R and Ball package).
    
    Parameters
    ----------
    top_k : int
        Number of features to select
    script_path : str, optional
        Path to R script (auto-detected if None)
    bcor_weight : str
        Weight method: 'chisquare', 'constant', or 'probability'
    n_jobs : int
        Number of parallel jobs
    """
    
    def __init__(self, top_k=100, script_path=None, 
                 bcor_weight="chisquare", n_jobs=8):
        self.top_k = top_k
        self.bcor_weight = bcor_weight
        self.n_jobs = n_jobs
        
        if script_path is None:
            self.script_path = str(Path(__file__).parent / "bcor_script.R")
        else:
            self.script_path = script_path
        
        self.scores_ = None
        self.selected_indices_ = None
    
    def fit(self, X, y):
        """Compute BCOR scores."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"R script not found: {self.script_path}")
        
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
                    self.bcor_weight, str(self.n_jobs)
                ], check=True, capture_output=True, timeout=3600)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"R script failed: {e.stderr.decode()}")
            except FileNotFoundError:
                raise RuntimeError("Rscript not found. Install R first.")
            
            if not os.path.exists(output_file):
                raise RuntimeError("R script did not produce output")
            
            result = pd.read_csv(output_file)
            self.scores_ = result[self.bcor_weight].values
        
        self.selected_indices_ = np.argsort(self.scores_)[-self.top_k:]
        return self
    
    def transform(self, X):
        """Select features."""
        return np.asarray(X)[:, self.selected_indices_]
    
    def fit_transform(self, X, y):
        """Fit and transform."""
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices=True):
        """Get selected features."""
        if indices:
            return self.selected_indices_
        mask = np.zeros(len(self.scores_), dtype=bool)
        mask[self.selected_indices_] = True
        return mask
    
    def get_scores(self):
        """Get all scores."""
        return self.scores_
