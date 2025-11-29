"""
Multi-Head Feature Selector

运行多个filter方法对应的Adaptive Lasso头，并收集所有结果。
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime
import warnings

from .feature_selector import AdaptiveLassoSelector

warnings.filterwarnings("ignore")


class MultiHeadSelector:
    """
    多头特征选择器。
    
    自动运行多个filter方法（SIS/Kendall/BCOR）及其对应的Adaptive Lasso头。
    
    Parameters
    ----------
    filter_methods : list of str, default=['sis', 'kendall']
        要使用的filter方法列表
    n_classes : int
        分类任务的类别数，回归任务设为1
    task_type : str, default='classification'
        任务类型: 'classification' 或 'regression'
    bcor_script_path : str, optional
        BCOR的R脚本路径（仅当使用bcor时需要）
    device : str, default='auto'
        计算设备: 'auto', 'cuda', 'cpu'
    **head_params
        传递给AdaptiveLassoSelector的其他参数
    
    Attributes
    ----------
    results_ : dict
        每个头的训练结果
    
    Examples
    --------
    >>> from mafs.models import MultiHeadSelector
    >>> from sklearn.model_selection import train_test_split
    >>> import numpy as np
    >>> 
    >>> # 生成数据
    >>> X = np.random.randn(200, 1000)
    >>> y = np.random.randint(0, 2, 200)
    >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    >>> 
    >>> # 训练多个头
    >>> multi_head = MultiHeadSelector(
    ...     filter_methods=['sis', 'kendall'],
    ...     n_classes=2,
    ...     task_type='classification'
    ... )
    >>> 
    >>> results = multi_head.fit(X_train, y_train, X_val, y_val, epochs=50)
    >>> 
    >>> # 查看每个头的结果
    >>> for method, result in results.items():
    ...     print(f"{method}: {len(result['selected_features'])} features")
    >>> 
    >>> # 获取用于聚合的特征
    >>> all_features = multi_head.get_all_selected_features(top_k_per_head=300)
    """
    
    def __init__(self,
                 filter_methods: List[str] = ['sis', 'kendall'],
                 n_classes: int = 2,
                 task_type: str = 'classification',
                 bcor_script_path: Optional[str] = None,
                 device: str = 'auto',
                 **head_params):
        
        self.filter_methods = filter_methods
        self.n_classes = n_classes
        self.task_type = task_type
        self.bcor_script_path = bcor_script_path
        self.device = device
        self.head_params = head_params
        
        # 验证filter方法
        valid_methods = ['sis', 'kendall', 'bcor']
        for method in filter_methods:
            if method not in valid_methods:
                raise ValueError(f"未知的filter方法: {method}. 有效方法: {valid_methods}")
        
        # 如果使用bcor但没有提供脚本路径，发出警告
        if 'bcor' in filter_methods and bcor_script_path is None:
            warnings.warn("使用BCOR方法但未提供R脚本路径，请设置bcor_script_path参数")
        
        self.results_ = None
        self.filter_objects_ = {}
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            lr: float = 0.001,
            weight_decay: float = 1e-5,
            reg_lambda: float = 1e-2,
            gamma: float = 0.5,
            patience: int = 10,
            verbose: bool = True,
            random_state: int = 42,
            save_dir: Optional[str] = None) -> Dict:
        """
        训练所有头。
        
        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            训练数据
        y_train : np.ndarray of shape (n_samples,)
            训练标签
        X_val : np.ndarray, optional
            验证数据
        y_val : np.ndarray, optional
            验证标签
        epochs : int, default=100
            训练轮数
        batch_size : int, default=32
            批次大小
        lr : float, default=0.001
            学习率
        weight_decay : float, default=1e-5
            权重衰减
        reg_lambda : float, default=1e-2
            Adaptive Lasso正则化系数
        gamma : float, default=0.5
            Adaptive Lasso的gamma参数
        patience : int, default=10
            早停耐心值
        verbose : bool, default=True
            是否打印训练信息
        random_state : int, default=42
            随机种子
        save_dir : str, optional
            保存每个头的模型的目录
            
        Returns
        -------
        results : dict
            每个头的结果，格式：
            {
                'sis': {
                    'filter_weights': array,
                    'final_weights': array,
                    'selected_features': array,
                    'model': AdaptiveLassoSelector,
                    'performance': {'val_loss': float, 'val_acc': float},
                    'training_time': float
                },
                ...
            }
        """
        start_time = datetime.now()
        
        if verbose:
            print("="*80)
            print("多头特征选择训练")
            print("="*80)
            print(f"Filter方法: {self.filter_methods}")
            print(f"训练数据: {X_train.shape}")
            if X_val is not None:
                print(f"验证数据: {X_val.shape}")
            print(f"任务类型: {self.task_type}")
            print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
        
        results = {}
        
        # For loop运行每个filter + 对应的头
        for i, method in enumerate(self.filter_methods, 1):
            method_start_time = time.time()
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"[{i}/{len(self.filter_methods)}] 训练 {method.upper()} 头")
                print(f"{'='*80}")
            
            try:
                # Step 1: 计算filter权重
                if verbose:
                    print(f"\nStep 1: 计算 {method.upper()} filter权重...")
                
                if method == 'sis':
                    from ..filters import SISFilter
                    filter_obj = SISFilter(n_jobs=8)
                    filter_obj.fit(X_train, y_train)
                    initial_weights = filter_obj.get_scores()
                    
                elif method == 'kendall':
                    from ..filters import KendallFilter
                    filter_obj = KendallFilter(n_jobs=8)
                    filter_obj.fit(X_train, y_train)
                    initial_weights = filter_obj.get_scores()
                    
                elif method == 'bcor':
                    from ..filters import BCORFilter
                    if self.bcor_script_path is None:
                        raise ValueError("使用BCOR方法需要提供bcor_script_path参数")
                    
                    filter_obj = BCORFilter(
                        script_path=self.bcor_script_path,
                        n_jobs=8
                    )
                    # BCOR需要额外参数
                    filter_obj.fit(
                        X_train, y_train,
                        weights_path='./temp_bcor_weights',
                        dataset_name='temp',
                        data_type='data',
                        y_type='target'
                    )
                    initial_weights = filter_obj.get_scores()
                
                # 保存filter对象
                self.filter_objects_[method] = filter_obj
                
                if verbose:
                    print(f"{method.upper()} filter权重统计:")
                    print(f"  最大值: {np.max(initial_weights):.6f}")
                    print(f"  最小值: {np.min(initial_weights):.6f}")
                    print(f"  平均值: {np.mean(initial_weights):.6f}")
                
                # Step 2: 用filter权重训练Adaptive Lasso头
                if verbose:
                    print(f"\nStep 2: 训练 {method.upper()} Adaptive Lasso头...")
                
                head = AdaptiveLassoSelector(
                    initial_weights=initial_weights,
                    n_classes=self.n_classes,
                    task_type=self.task_type,
                    **self.head_params
                )
                
                head.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    patience=patience,
                    device=self.device,
                    verbose=verbose,
                    random_state=random_state
                )
                
                # Step 3: 保存结果
                final_weights = head.get_final_weights()
                selected_features = head.get_selected_features()
                
                method_time = time.time() - method_start_time
                
                results[method] = {
                    'filter_weights': initial_weights,
                    'final_weights': final_weights,
                    'selected_features': selected_features,
                    'model': head,
                    'performance': {
                        'val_loss': head.best_val_loss,
                        'val_acc': head.best_val_acc
                    },
                    'training_time': method_time,
                    'filter_object': filter_obj
                }
                
                if verbose:
                    print(f"\n{method.upper()} 头训练完成！")
                    print(f"  训练时间: {method_time:.2f}秒")
                    print(f"  最佳验证损失: {head.best_val_loss:.4f}")
                    print(f"  最佳验证准确率: {head.best_val_acc:.4f}")
                    print(f"  选择的特征数: {len(selected_features)}")
                    print(f"  Top 10 特征: {selected_features[:10].tolist()}")
                
                # 可选: 保存模型
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(save_dir, f'{method}_head.pt')
                    head.save(model_path)
                    if verbose:
                        print(f"  模型已保存到: {model_path}")
                
            except Exception as e:
                print(f"\n错误: 训练 {method.upper()} 头时失败")
                print(f"错误信息: {str(e)}")
                import traceback
                traceback.print_exc()
                
                results[method] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.results_ = results
        
        total_time = datetime.now() - start_time
        
        if verbose:
            print(f"\n{'='*80}")
            print("多头训练完成!")
            print(f"{'='*80}")
            print(f"总耗时: {total_time.total_seconds():.2f}秒")
            
            # 打印汇总表
            print(f"\n结果汇总:")
            print(f"{'方法':<10} {'验证损失':<12} {'验证准确率':<12} {'训练时间(秒)':<15} {'状态'}")
            print("-" * 60)
            
            for method in self.filter_methods:
                if method in results:
                    result = results[method]
                    if 'status' in result and result['status'] == 'failed':
                        print(f"{method:<10} {'N/A':<12} {'N/A':<12} {'N/A':<15} 失败")
                    else:
                        val_loss = result['performance']['val_loss']
                        val_acc = result['performance']['val_acc']
                        train_time = result['training_time']
                        print(f"{method:<10} {val_loss:<12.4f} {val_acc:<12.4f} {train_time:<15.2f} 成功")
            print("="*80)
        
        return results
    
    def get_all_selected_features(self, 
                                   top_k_per_head: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        获取所有头选择的特征（用于后续聚合）。
        
        Parameters
        ----------
        top_k_per_head : int, optional
            每个头取top_k个特征，如果为None则取所有
            
        Returns
        -------
        dict
            每个头选择的特征索引
            格式: {'sis': array([...]), 'kendall': array([...]), ...}
        """
        if self.results_ is None:
            raise ValueError("必须先调用fit()方法")
        
        all_features = {}
        
        for method, result in self.results_.items():
            if 'selected_features' in result:
                features = result['selected_features']
                if top_k_per_head is not None:
                    features = features[:top_k_per_head]
                all_features[method] = features
        
        return all_features
    
    def get_performance_summary(self) -> Dict:
        """
        获取所有头的性能汇总。
        
        Returns
        -------
        dict
            性能汇总信息
        """
        if self.results_ is None:
            raise ValueError("必须先调用fit()方法")
        
        summary = {}
        
        for method, result in self.results_.items():
            if 'performance' in result:
                summary[method] = result['performance']
        
        return summary
    
    def save_all_results(self, save_dir: str):
        """
        保存所有头的结果。
        
        Parameters
        ----------
        save_dir : str
            保存目录
        """
        if self.results_ is None:
            raise ValueError("必须先调用fit()方法")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for method, result in self.results_.items():
            if 'model' in result:
                # 保存模型
                model_path = os.path.join(save_dir, f'{method}_model.pt')
                result['model'].save(model_path)
                
                # 保存特征索引
                features_path = os.path.join(save_dir, f'{method}_features.npy')
                np.save(features_path, result['selected_features'])
                
                # 保存权重
                weights_path = os.path.join(save_dir, f'{method}_weights.npy')
                np.save(weights_path, result['final_weights'])
                
                print(f"{method} 结果已保存到 {save_dir}")
    
    def get_feature_importance_comparison(self, top_k: int = 20) -> Dict:
        """
        比较不同头选择的top特征。
        
        Parameters
        ----------
        top_k : int, default=20
            比较top_k个特征
            
        Returns
        -------
        dict
            每个头的top特征
        """
        if self.results_ is None:
            raise ValueError("必须先调用fit()方法")
        
        comparison = {}
        
        for method, result in self.results_.items():
            if 'selected_features' in result:
                comparison[method] = result['selected_features'][:top_k].tolist()
        
        return comparison
