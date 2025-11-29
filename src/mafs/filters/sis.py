"""Sure Independence Screening filter."""

import numpy as np
from scipy import stats
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")


def _sis_worker(args):
    """Compute Pearson correlation for one feature."""
    feature_data, label, idx = args
    try:
        r, _ = stats.pearsonr(feature_data, label)
        return idx, (0 if np.isnan(r) else abs(r))
    except:
        return idx, 0


class SISFilter:
    """
    Sure Independence Screening using Pearson correlation.
    
    Parameters
    ----------
    top_k : int
        Number of features to select
    n_jobs : int
        Number of parallel jobs
    """
    
    def __init__(self, top_k=100, n_jobs=8):
        self.top_k = top_k
        self.n_jobs = n_jobs
        self.scores_ = None
        self.selected_indices_ = None
    
    def fit(self, X, y):
        """Compute SIS scores."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        args_list = [(X[:, i], y, i) for i in range(X.shape[1])]
        
        self.scores_ = np.zeros(X.shape[1])
        with Pool(self.n_jobs) as pool:
            for idx, score in pool.map(_sis_worker, args_list):
                self.scores_[idx] = score
        
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
