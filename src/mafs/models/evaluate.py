"""
Feature Selection Evaluator
Evaluates feature selection results against ground truth
"""

import numpy as np
import os


class FeatureSelectionEvaluator:
    """Evaluator for feature selection with ground truth"""
    
    def __init__(self, data_file_path):
        """
        Initialize evaluator
        
        Args:
            data_file_path: Path to .npz file containing 'variables' field
        """
        self.data_file_path = data_file_path
        self.ground_truth = None
        self.load_ground_truth()
    
    def load_ground_truth(self):
        """Load ground truth feature groups from data file"""
        if not os.path.exists(self.data_file_path):
            print(f"Warning: Data file not found: {self.data_file_path}")
            return
        
        loaded_data = np.load(self.data_file_path, allow_pickle=True)
        
        if 'variables' not in loaded_data:
            print(f"Warning: No 'variables' field in {self.data_file_path}")
            return
        
        variables = loaded_data['variables']
        n_groups = len(variables)
        
        self.ground_truth = {}
        
        if n_groups == 14:
            # Combine type: 7 numerical + 7 categorical
            print(f"Detected COMBINE type: {n_groups} groups")
            relationship_types = ["LINEAR", "COS", "LOG", "POWER", "EXP", "COMBINE", "COS_EXP_COMBINE"]
            
            # Merge NUM and CAT
            for i, rel_type in enumerate(relationship_types):
                num_features = variables[i].tolist() if isinstance(variables[i], np.ndarray) else list(variables[i])
                cat_features = variables[i+7].tolist() if isinstance(variables[i+7], np.ndarray) else list(variables[i+7])
                self.ground_truth[rel_type] = num_features + cat_features
                
        elif n_groups == 7:
            # Single type: 7 groups
            print(f"Detected SINGLE type: {n_groups} groups")
            relationship_types = ["LINEAR", "COS", "LOG", "POWER", "EXP", "COMBINE", "COS_EXP_COMBINE"]
            
            for i, rel_type in enumerate(relationship_types):
                features = variables[i].tolist() if isinstance(variables[i], np.ndarray) else list(variables[i])
                self.ground_truth[rel_type] = features
        else:
            print(f"Warning: Unknown number of groups ({n_groups})")
            return
        
        print(f"\nGround truth loaded:")
        for rel_type, features in self.ground_truth.items():
            print(f"  {rel_type}: {len(features)} features")
    
    def evaluate(self, selected_features, top_k_list=[100, 200, 300, 500]):
        if self.ground_truth is None:
            print("No ground truth available")
            return None
        
        results = {}
        
        for top_k in top_k_list:
            top_features = selected_features[:top_k]
            top_set = set(top_features)
            
            results[f'top_{top_k}'] = {}
            
            for rel_type, true_features in self.ground_truth.items():
                true_set = set(true_features)
                selected_count = len(top_set & true_set)
                total_count = len(true_features)
                
                results[f'top_{top_k}'][rel_type] = {
                    'selected': selected_count,
                    'total': total_count,
                    'recall': selected_count / total_count if total_count > 0 else 0,
                    'selected_features': sorted(list(top_set & true_set))
                }
        
        return results
    
    def print_results(self, results, title="EVALUATION RESULTS"):
        if results is None:
            print("No results to print")
            return
        
        print("\n" + "="*80)
        print(title)
        print("="*80)
        
        for top_k, rel_results in results.items():
            print(f"\n{'='*80}")
            print(f"{top_k.upper().replace('_', ' ')}")
            print("="*80)
            
            # Summary table
            print(f"\n{'Relationship':<20} {'Selected':<10} {'Total':<10} {'Recall':<10}")
            print("-" * 60)
            
            for rel_type, metrics in rel_results.items():
                recall_pct = metrics['recall'] * 100
                print(f"{rel_type:<20} {metrics['selected']:<10} {metrics['total']:<10} {recall_pct:>6.1f}%")
            
            # Overall statistics
            total_selected = sum(m['selected'] for m in rel_results.values())
            total_true = sum(m['total'] for m in rel_results.values())
            overall_recall = total_selected / total_true if total_true > 0 else 0
            
            print("-" * 60)
            print(f"{'OVERALL':<20} {total_selected:<10} {total_true:<10} {overall_recall*100:>6.1f}%")
            
            # Best and worst
            recalls = [(rel, m['recall']) for rel, m in rel_results.items()]
            best = max(recalls, key=lambda x: x[1])
            worst = min(recalls, key=lambda x: x[1])
            
            print(f"\n  Best:  {best[0]:<20} ({best[1]*100:.1f}%)")
            print(f"  Worst: {worst[0]:<20} ({worst[1]*100:.1f}%)")
    
    def get_summary(self, results):
        if results is None:
            return None
        
        summary = {}
        
        for top_k, rel_results in results.items():
            total_selected = sum(m['selected'] for m in rel_results.values())
            total_true = sum(m['total'] for m in rel_results.values())
            overall_recall = total_selected / total_true if total_true > 0 else 0
            
            summary[top_k] = {
                'overall_recall': overall_recall,
                'total_selected': total_selected,
                'total_true': total_true
            }
        
        return summary


def evaluate_features(selected_features, data_file_path, top_k_list=[100, 200, 300, 500], print_output=True):
    evaluator = FeatureSelectionEvaluator(data_file_path)
    results = evaluator.evaluate(selected_features, top_k_list)
    
    if print_output and results is not None:
        evaluator.print_results(results)
    
    return results


