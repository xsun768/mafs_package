"""
Test MultiHeadEARFS with real simulation data
Simplified version with minimal parameters
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse

# Add paths
models_dir = os.path.join(os.path.dirname(__file__), 'models')
if os.path.exists(models_dir):
    sys.path.insert(0, models_dir)
else:
    sys.path.insert(0, os.path.dirname(__file__))

print("="*60)
print("MultiHeadEARFS Test")
print("="*60)

# Import modules
try:
    from multi_head_earfs import MultiHeadEARFS
    print("✓ Successfully imported MultiHeadEARFS")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def setup_seed(seed=42):
    """Set random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(file_path, y_type='numerical', test_size=0.2, seed=42):
    """Load and prepare dataset"""
    setup_seed(seed)
    
    print(f"\nLoading data from {file_path}")
    loaded_data = np.load(file_path, allow_pickle=True)
    all_data = loaded_data['X']
    all_label = loaded_data['Y']
    
    print(f"Total samples: {len(all_data)}")
    print(f"Features: {all_data.shape[1]}")
    
    # Process labels
    if y_type == 'categorical':
        all_label = all_label.astype(int)
        print(f"Label range: {np.min(all_label)} to {np.max(all_label)}")
        num_label = int(np.max(all_label) + 1)
    else:
        all_label = all_label.astype(float)
        print(f"Label stats: min={np.min(all_label):.4f}, max={np.max(all_label):.4f}")
        num_label = 1
    
    # Split
    if y_type == 'categorical' and len(np.unique(all_label)) > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            all_data, all_label, test_size=test_size, 
            random_state=seed, stratify=all_label
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            all_data, all_label, test_size=test_size, random_state=seed
        )
    
    # Normalize Y for regression
    if y_type == 'numerical':
        y_mean, y_std = np.mean(y_train), np.std(y_train)
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
    
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    
    # Return dictionary instead of tuple
    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'n_feature': all_data.shape[1],
        'num_label': num_label
    }

def create_weight_files(n_features, methods, weight_dir='./test_weights'):
    """Create dummy weight files"""
    os.makedirs(weight_dir, exist_ok=True)
    
    weight_files = []
    
    for method in methods:
        weight_file = os.path.join(weight_dir, f'{method}_weights.csv')
        weights = np.random.rand(n_features)
        pd.DataFrame({method: weights}).to_csv(weight_file, index=False)
        weight_files.append(weight_file)
    
    print(f"\n✓ Created {len(weight_files)} weight files: {methods}")
    return weight_files

def test_dataset(data_path, y_type, reg_lambda,device, hidden_scale, methods):
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
    weight_files = create_weight_files(n_features, methods)
    
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
    
    # Test MultiHeadEARFS with evaluation enabled
    print("\nInitializing MultiHeadEARFS with evaluator...")
    try:
        multi_head = MultiHeadEARFS(
            input_size=n_features,
            n_classes=n_classes,
            weight_files=weight_files,
            hidden_scale=hidden_scale,
            dropout_rate=0.4,
            y_type=y_type,
            device=device,
            data_file_path=data_path,        
            reg_lambda=reg_lambda     
        )
    
        print("MultiHeadEARFS initialized")
        
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

    
def parse_args():
    parser = argparse.ArgumentParser(description="Test MultiHeadEARFS")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .npz dataset file"
    )
    parser.add_argument(
        "--y_type",
        type=str,
        default="categorical",
        choices=["categorical", "numerical"],
        help="Type of labels"
    )
    parser.add_argument(
        "--hidden_scale",
        type=int,
        default=200,
        help="Hidden scale for classifier"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        default=['sis', 'bcor', 'kendall'],
        choices=['sis', 'bcor', 'kendall'],
        help="Weight methods to use"
    )
    
    # EARFS specific arguments
    parser.add_argument('--reg_lambda', type=float, default=1e-2,help='EARFS regularization lambda')
    
    return parser.parse_args()



    

def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return False
    
    print(f"\nData: {args.data_path}")
    print(f"Task: {args.y_type}")
    print(f"Hidden scale: {args.hidden_scale}")
    print(f"Methods: {args.methods}")
    
    success = test_dataset(
        args.data_path,
        args.y_type, 
        args.reg_lambda,
        device,
        args.hidden_scale,
        args.methods
    )
    
    print(f"\nResult: {'Finished' if success else 'FAILED'}")
    return success



if __name__ == "__main__":
    main()
