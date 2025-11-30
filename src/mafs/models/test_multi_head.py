"""
Test MultiHeadSelector with real simulation data
Tests both numerical and categorical datasets
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


models_dir = os.path.join(os.path.dirname(__file__), 'models')
if os.path.exists(models_dir):
    sys.path.insert(0, models_dir)
else:
    sys.path.insert(0, os.path.dirname(__file__))

print("="*60)
print("MultiHeadSelector Test - Real Data")
print("="*60)


try:
    import multi_head
    from multi_head import MultiHeadSelector, train_multi_head
    print("Successfully imported MultiHeadSelector")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def setup_seed(seed):
    """Set random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(file_path, seed, y_type='numerical', test_size=0.2):
    """Load and prepare dataset"""
    setup_seed(seed)
    
    print(f"\nLoading data from {file_path}")
    loaded_data = np.load(file_path, allow_pickle=True)
    all_data = loaded_data['X']
    all_label = loaded_data['Y']
    
    total_samples = len(all_data)
    print(f"Total samples: {total_samples}")
    print(f"Features: {all_data.shape[1]}")
    
    # Process labels based on task type
    if y_type == 'categorical':
        all_label = all_label.astype(int)
        print(f"Label range: {np.min(all_label)} to {np.max(all_label)}")
        num_label = int(np.max(all_label) + 1)
    else:
        all_label = all_label.astype(float)
        print(f"Label stats: min={np.min(all_label):.4f}, max={np.max(all_label):.4f}, mean={np.mean(all_label):.4f}")
        num_label = 1
    
    # Split train/val
    if y_type == 'categorical' and len(np.unique(all_label)) > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            all_data, all_label, 
            test_size=test_size, 
            random_state=seed,
            stratify=all_label
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            all_data, all_label, 
            test_size=test_size, 
            random_state=seed
        )
    
    # Normalize Y for regression
    y_scaler_params = None
    if y_type == 'numerical':
        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)
        
        y_train = (y_train - y_train_mean) / y_train_std
        y_val = (y_val - y_train_mean) / y_train_std
        
        y_scaler_params = {'mean': y_train_mean, 'std': y_train_std}
    
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'n_feature': X_train.shape[1],
        'num_label': num_label,
        'y_scaler_params': y_scaler_params
    }


def create_weight_files(n_features, weight_dir='./test_weights'):
    """Create dummy weight files for testing"""
    os.makedirs(weight_dir, exist_ok=True)
    
    weight_files = []
    weight_types = ['sis', 'kendall', 'bcor']
    
    for wtype in weight_types:
        weight_file = os.path.join(weight_dir, f'{wtype}_weights.csv')
        
        # Generate random weights
        weights = np.random.rand(n_features)
        
        pd.DataFrame({wtype: weights}).to_csv(weight_file, index=False)
        weight_files.append(weight_file)
    
    print(f"\n✓ Created {len(weight_files)} weight files")
    return weight_files


def test_dataset(data_path, y_type, device):
    """Test MultiHeadSelector on a dataset"""
    print("\n" + "="*60)
    print(f"Testing {y_type.upper()} dataset")
    print("="*60)
    
    # Load data
    data = prepare_dataset(data_path, seed=42, y_type=y_type)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    n_features = data['n_feature']
    n_classes = data['num_label']
    
    # Create weight files
    weight_files = create_weight_files(n_features)
    
    # Create DataLoaders
    if y_type == 'categorical':
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
    else:
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).unsqueeze(1)
        )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("\n✓ DataLoaders created")
    
    # Test MultiHeadSelector with evaluation enabled
    print("\nInitializing MultiHeadSelector with evaluator...")
    try:
        multi_head = MultiHeadSelector(
            input_size=n_features,
            n_classes=n_classes,
            weight_files=weight_files,
            hidden_scale=4,
            dropout_rate=0.2,
            y_type=y_type,
            device=device,
            data_file_path=data_path  # ← Enable evaluation!
        )
        print("MultiHeadSelector initialized")
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Train
    print("\nTraining...")
    try:
        results = multi_head.fit(train_loader, val_loader)
        
        print(f"\n✓ Training completed")
        print(f"Trained {len(results)} heads")
        
        # Show results
        for i, result in enumerate(results):
            weights = result['weights']
            top_features = result['top_features']
            
            print(f"\nHead {i}:")
            print(f"  Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
            print(f"  Top 10 features: {top_features[:10]}")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    import shutil
    if os.path.exists('./test_weights'):
        shutil.rmtree('./test_weights')
    
    return True




def main():
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Data paths
    data_dir = '../../../data/simulation_data'

    #################################################################################################
    # numerical_data = os.path.join(data_dir, 'data_25k_combine_numerical.npz')

    # if not os.path.exists(numerical_data):
    #     print(f"\n✗ Numerical data not found: {numerical_data}")
    #     return False
    # numerical_success = test_dataset(numerical_data, 'numerical', device)
    # print(f"Numerical dataset: {'Finished' if numerical_success else 'FAILED'}")
    #################################################################################################

    categorical_data = os.path.join(data_dir, 'data_25k_combine_categorical.npz')
    if not os.path.exists(categorical_data):
        print(f"\n✗ Categorical data not found: {categorical_data}")
        return False
    categorical_success = test_dataset(categorical_data, 'categorical', device)
    print(f"Categorical dataset: {'Finished' if categorical_success else 'FAILED'}")
    
    #################################################################################################


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
