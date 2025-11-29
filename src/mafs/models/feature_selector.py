"""
Adaptive Lasso Feature Selector

单个特征选择头的实现，使用Adaptive Lasso进行特征选择。
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from typing import Union, Optional, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


class FeatureSelection(nn.Module):
    """
    特征选择层，使用初始权重进行初始化。
    
    Parameters
    ----------
    initial_weights : array-like of shape (n_features,)
        来自filter方法的初始权重
    normalize : bool, default=True
        是否对初始权重进行z-score标准化
    """
    
    def __init__(self, initial_weights: Union[np.ndarray, torch.Tensor], 
                 normalize: bool = True):
        super().__init__()
        
        # 转换为tensor
        if isinstance(initial_weights, np.ndarray):
            initial_weights = torch.tensor(initial_weights, dtype=torch.float32)
        
        # 确保是1D或2D
        if len(initial_weights.shape) == 1:
            initial_weights = initial_weights.unsqueeze(0)
        
        self.initial_weights = initial_weights
        
        print("初始权重统计:")
        print(f"  最大值: {torch.max(initial_weights).item():.6f}")
        print(f"  最小值: {torch.min(initial_weights).item():.6f}")
        print(f"  平均值: {torch.mean(initial_weights).item():.6f}")
        print(f"  标准差: {torch.std(initial_weights).item():.6f}")
        
        # 可选的标准化
        if normalize:
            normalized_weights = self._z_score_normalize(initial_weights)
            print("\n标准化后权重统计:")
            print(f"  最大值: {torch.max(normalized_weights).item():.6f}")
            print(f"  最小值: {torch.min(normalized_weights).item():.6f}")
            print(f"  平均值: {torch.mean(normalized_weights).item():.6f}")
            print(f"  标准差: {torch.std(normalized_weights).item():.6f}")
            self.selection_rate = nn.Parameter(normalized_weights)
        else:
            self.selection_rate = nn.Parameter(initial_weights)
    
    def _z_score_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score标准化"""
        mu = torch.mean(x)
        sigma = torch.std(x, unbiased=False)
        
        if sigma < 1e-8:
            print(f"警告: 标准差过小 ({sigma})，跳过标准化")
            return x
        
        normalized = (x - mu) / sigma
        return normalized
    
    def get_selection_rate(self) -> torch.Tensor:
        """获取当前的特征选择权重（使用ReLU保证非负）"""
        return torch.relu(self.selection_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对输入特征加权
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_features)
            输入特征
            
        Returns
        -------
        torch.Tensor of shape (batch_size, n_features)
            加权后的特征
        """
        return x * torch.relu(self.selection_rate)


class Classifier(nn.Module):
    """
    分类器/回归器网络。
    
    Parameters
    ----------
    input_size : int
        输入特征数
    n_classes : int
        输出类别数（分类）或1（回归）
    hidden_scale : int, default=4
        隐藏层大小的缩放因子
    dropout_rate : float, default=0.2
        Dropout比率
    activation : str, default='relu'
        激活函数类型
    task_type : str, default='classification'
        任务类型: 'classification' 或 'regression'
    """
    
    def __init__(self, 
                 input_size: int,
                 n_classes: int,
                 hidden_scale: int = 4,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 task_type: str = 'classification'):
        super().__init__()
        
        self.task_type = task_type
        
        # 激活函数映射
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'silu': nn.SiLU()
        }
        
        # 计算隐藏层大小
        hidden_size = int(input_size / hidden_scale)
        print(f"Classifier hidden size: {hidden_size}")
        
        activation_fn = activation_functions.get(activation.lower(), nn.ReLU())
        
        # 构建网络
        layers = [
            # 第一层
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation_fn,
            nn.Dropout(p=dropout_rate),
            
            # 第二层
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation_fn,
            nn.Dropout(p=dropout_rate),
            
            # 输出层
            nn.Linear(hidden_size, n_classes if task_type == 'classification' else 1)
        ]
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AdaptiveLassoSelector(nn.Module):
    """
    Adaptive Lasso特征选择器。
    
    结合filter方法的初始权重和神经网络训练，使用Adaptive Lasso进行特征选择。
    
    Parameters
    ----------
    initial_weights : array-like of shape (n_features,)
        来自filter方法（SIS/Kendall/BCOR）的初始权重
    n_classes : int
        分类任务的类别数，回归任务设为1
    task_type : str, default='classification'
        任务类型: 'classification' 或 'regression'
    hidden_scale : int, default=4
        分类器隐藏层缩放因子
    dropout_rate : float, default=0.2
        Dropout比率
    activation : str, default='relu'
        激活函数
    normalize_weights : bool, default=True
        是否标准化初始权重
    
    Attributes
    ----------
    feature_selection : FeatureSelection
        特征选择层
    classifier : Classifier
        分类器/回归器
    best_val_loss : float
        最佳验证损失
    best_val_acc : float
        最佳验证准确率
    
    Examples
    --------
    >>> from mafs.models import AdaptiveLassoSelector
    >>> from mafs.filters import SISFilter
    >>> import numpy as np
    >>> 
    >>> # 生成数据
    >>> X = np.random.randn(100, 1000)
    >>> y = np.random.randint(0, 2, 100)
    >>> 
    >>> # 计算SIS权重
    >>> sis = SISFilter()
    >>> sis.fit(X, y)
    >>> 
    >>> # 训练Adaptive Lasso选择器
    >>> selector = AdaptiveLassoSelector(
    ...     initial_weights=sis.get_scores(),
    ...     n_classes=2,
    ...     task_type='classification'
    ... )
    >>> selector.fit(X, y, epochs=50)
    >>> 
    >>> # 获取选择的特征
    >>> selected = selector.get_selected_features(top_k=100)
    """
    
    def __init__(self,
                 initial_weights: Union[np.ndarray, torch.Tensor],
                 n_classes: int,
                 task_type: str = 'classification',
                 hidden_scale: int = 4,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 normalize_weights: bool = True):
        super().__init__()
        
        self.task_type = task_type
        self.n_classes = n_classes
        
        # 确定输入大小
        if isinstance(initial_weights, np.ndarray):
            n_features = len(initial_weights)
        else:
            n_features = initial_weights.shape[-1]
        
        self.n_features = n_features
        
        # 初始化特征选择层
        self.feature_selection = FeatureSelection(
            initial_weights=initial_weights,
            normalize=normalize_weights
        )
        
        # 初始化分类器
        self.classifier = Classifier(
            input_size=n_features,
            n_classes=n_classes,
            hidden_scale=hidden_scale,
            dropout_rate=dropout_rate,
            activation=activation,
            task_type=task_type
        )
        
        # 训练结果记录
        self.best_val_loss = None
        self.best_val_acc = None
        self.history = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_features)
            输入数据
            
        Returns
        -------
        torch.Tensor
            预测输出
        """
        x_selected = self.feature_selection(x)
        return self.classifier(x_selected)
    
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
            device: str = 'auto',
            verbose: bool = True,
            random_state: int = 42) -> 'AdaptiveLassoSelector':
        """
        训练Adaptive Lasso选择器。
        
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
        device : str, default='auto'
            设备: 'auto', 'cuda', 'cpu'
        verbose : bool, default=True
            是否打印训练信息
        random_state : int, default=42
            随机种子
            
        Returns
        -------
        self : AdaptiveLassoSelector
            训练后的选择器
        """
        # 设置随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # 设置设备
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.to(self.device)
        
        if verbose:
            print(f"使用设备: {self.device}")
            print(f"训练数据: {X_train.shape}, 验证数据: {X_val.shape if X_val is not None else 'None'}")
        
        # 准备数据
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        if self.task_type == 'classification':
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        
        train_dataset = Data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 验证数据
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            if self.task_type == 'classification':
                y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            else:
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            
            val_dataset = Data.TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # 计算Adaptive Lasso惩罚系数tau
        initial_weights = self.feature_selection.initial_weights.to(self.device)
        if len(initial_weights.shape) > 1:
            initial_weights = initial_weights.squeeze(0)
        
        epsilon = 1e-5
        tau = 1.0 / (torch.abs(initial_weights) + epsilon) ** gamma
        tau = torch.clamp(tau, max=20.0)
        
        if verbose:
            print(f"\nAdaptive Lasso惩罚系数tau统计:")
            print(f"  最大值: {torch.max(tau).item():.4f}")
            print(f"  最小值: {torch.min(tau).item():.4f}")
            print(f"  平均值: {torch.mean(tau).item():.4f}")
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=epochs, T_mult=1, eta_min=1e-3
        )
        
        if self.task_type == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': []
        }
        
        # 最佳模型追踪
        best_val_loss = float('inf')
        best_val_metric = 0 if self.task_type == 'classification' else float('inf')
        best_state = None
        counter = 0
        
        # 训练循环
        if verbose:
            print(f"\n开始训练 ({epochs} epochs)...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.train()
            train_loss = 0.0
            train_metric = 0.0
            n_train = 0
            
            for inputs, targets in train_loader:
                if inputs.size(0) <= 1:
                    continue
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                
                # Adaptive Lasso正则化
                selection_rate = self.feature_selection.get_selection_rate()
                reg_loss = reg_lambda * torch.sum(tau * torch.abs(selection_rate))
                total_loss = loss + reg_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                # 记录
                batch_size_actual = inputs.size(0)
                train_loss += total_loss.item() * batch_size_actual
                
                if self.task_type == 'classification':
                    preds = outputs.argmax(dim=1)
                    train_metric += (preds == targets).sum().item()
                else:
                    mae = torch.abs(outputs - targets).mean().item()
                    train_metric += (1.0 / (1.0 + mae)) * batch_size_actual
                
                n_train += batch_size_actual
            
            train_loss /= n_train
            train_metric /= n_train
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            
            # 验证阶段
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_metric = 0.0
                n_val = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self(inputs)
                        loss = loss_fn(outputs, targets)
                        
                        batch_size_actual = inputs.size(0)
                        val_loss += loss.item() * batch_size_actual
                        
                        if self.task_type == 'classification':
                            preds = outputs.argmax(dim=1)
                            val_metric += (preds == targets).sum().item()
                        else:
                            mae = torch.abs(outputs - targets).mean().item()
                            val_metric += (1.0 / (1.0 + mae)) * batch_size_actual
                        
                        n_val += batch_size_actual
                
                val_loss /= n_val
                val_metric /= n_val
                
                # 记录历史
                history['val_loss'].append(val_loss)
                history['val_metric'].append(val_metric)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_metric = val_metric
                    best_state = self.state_dict().copy()
                    counter = 0
                else:
                    counter += 1
                
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
                
                # 打印信息
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    metric_name = 'acc' if self.task_type == 'classification' else 'inv_mae'
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"lr: {current_lr:.6f} - "
                          f"loss: {train_loss:.4f} - "
                          f"{metric_name}: {train_metric:.4f} - "
                          f"val_loss: {val_loss:.4f} - "
                          f"val_{metric_name}: {val_metric:.4f}")
            else:
                # 没有验证集
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    metric_name = 'acc' if self.task_type == 'classification' else 'inv_mae'
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"lr: {current_lr:.6f} - "
                          f"loss: {train_loss:.4f} - "
                          f"{metric_name}: {train_metric:.4f}")
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_metric'].append(train_metric)
        
        # 加载最佳模型
        if best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                metric_name = 'accuracy' if self.task_type == 'classification' else 'inverse MAE'
                print(f"\n训练完成！")
                print(f"最佳验证损失: {best_val_loss:.4f}")
                print(f"最佳验证{metric_name}: {best_val_metric:.4f}")
        
        # 保存训练结果
        self.best_val_loss = best_val_loss
        self.best_val_acc = best_val_metric
        self.history = history
        
        return self
    
    def get_final_weights(self) -> np.ndarray:
        """
        获取训练后的最终特征权重。
        
        Returns
        -------
        np.ndarray of shape (n_features,)
            特征权重
        """
        weights = self.feature_selection.get_selection_rate()
        return weights.detach().cpu().numpy().flatten()
    
    def get_selected_features(self, top_k: Optional[int] = None) -> np.ndarray:
        """
        获取选择的特征索引（按重要性排序）。
        
        Parameters
        ----------
        top_k : int, optional
            返回top_k个特征，如果为None则返回所有特征
            
        Returns
        -------
        np.ndarray
            特征索引（降序排列）
        """
        weights = self.get_final_weights()
        sorted_indices = np.argsort(weights)[::-1]
        
        if top_k is not None:
            return sorted_indices[:top_k]
        return sorted_indices
    
    def transform(self, X: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        使用选择的特征转换数据。
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            要转换的数据
        top_k : int, optional
            使用top_k个特征
            
        Returns
        -------
        np.ndarray of shape (n_samples, top_k)
            转换后的数据
        """
        selected_indices = self.get_selected_features(top_k=top_k)
        return X[:, selected_indices]
    
    def save(self, filepath: str):
        """
        保存模型。
        
        Parameters
        ----------
        filepath : str
            保存路径
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'task_type': self.task_type,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'final_weights': self.get_final_weights()
        }
        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, **kwargs):
        """
        加载模型。
        
        Parameters
        ----------
        filepath : str
            模型文件路径
        **kwargs
            传递给模型初始化的其他参数
            
        Returns
        -------
        AdaptiveLassoSelector
            加载的模型
        """
        save_dict = torch.load(filepath, map_location='cpu')
        
        # 重建模型
        model = cls(
            initial_weights=save_dict['final_weights'],
            n_classes=save_dict['n_classes'],
            task_type=save_dict['task_type'],
            **kwargs
        )
        
        model.load_state_dict(save_dict['model_state_dict'])
        model.best_val_loss = save_dict.get('best_val_loss')
        model.best_val_acc = save_dict.get('best_val_acc')
        model.history = save_dict.get('history')
        
        print(f"模型已从 {filepath} 加载")
        return model
