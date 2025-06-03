"""
Hyperparameter optimization module for neural network tuning
"""

import itertools
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error
import torch
import time

class HyperparameterOptimizer:
    """
    Hyperparameter optimization using grid search or random search
    """
    
    def __init__(self, config_manager, model_trainer):
        self.config = config_manager
        self.trainer = model_trainer
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization
        """
        if not self.config.get('model.grid_search.enabled', False):
            self.logger.info("Grid search disabled. Using default parameters.")
            return self._get_default_params()
        
        self.logger.info("Starting hyperparameter optimization...")
        
        # Get parameter grid
        param_grid = self._create_parameter_grid()
        
        best_params = None
        best_score = float('inf')
        best_model = None
        
        results = []
        
        total_combinations = len(param_grid)
        self.logger.info(f"Testing {total_combinations} parameter combinations...")
        
        for i, params in enumerate(param_grid):
            start_time = time.time()
            
            # Update config with current parameters
            self._update_config_with_params(params)
            
            try:
                # Create and train model with current parameters
                model = self.trainer.create_model(X_train.shape[1])
                result = self.trainer.train_single_fold(model, X_train, y_train, X_val, y_val)
                
                score = result['best_val_loss']
                training_time = time.time() - start_time
                
                results.append({
                    'params': params.copy(),
                    'score': score,
                    'training_time': training_time
                })
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    best_model = result['model']
                
                self.logger.info(f"Combination {i+1}/{total_combinations}: "
                               f"Score = {score:.6f}, Time = {training_time:.2f}s")
                self.logger.debug(f"Parameters: {params}")
                
            except Exception as e:
                self.logger.error(f"Error training with parameters {params}: {e}")
                continue
        
        # Update config with best parameters
        if best_params:
            self._update_config_with_params(best_params)
            self.logger.info(f"Best parameters found:")
            for key, value in best_params.items():
                self.logger.info(f"  {key}: {value}")
            self.logger.info(f"Best validation score: {best_score:.6f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'all_results': results
        }
    
    def _create_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create parameter grid from config"""
        grid_params = self.config.get('model.grid_search.parameters', {})
        
        # Default grid if not specified
        if not grid_params:
            grid_params = {
                'learning_rate': [0.001, 0.01, 0.0001],
                'hidden_layers': [[64, 32], [128, 64, 32], [256, 128, 64]],
                'dropout_rate': [0.2, 0.3, 0.4],
                'batch_size': [16, 32, 64]
            }
        
        # Convert to parameter grid
        param_grid = list(ParameterGrid(grid_params))
        
        return param_grid
    
    def _update_config_with_params(self, params: Dict[str, Any]):
        """Update configuration with current parameters"""
        param_mapping = {
            'learning_rate': 'model.training.learning_rate',
            'hidden_layers': 'model.architecture.hidden_layers',
            'dropout_rate': 'model.architecture.dropout_rate',
            'batch_size': 'model.training.batch_size',
            'epochs': 'model.training.epochs'
        }
        
        for param_name, param_value in params.items():
            if param_name in param_mapping:
                config_path = param_mapping[param_name]
                self.config.set(config_path, param_value)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters from config"""
        return {
            'learning_rate': self.config.get('model.training.learning_rate', 0.001),
            'hidden_layers': self.config.get('model.architecture.hidden_layers', [128, 64, 32]),
            'dropout_rate': self.config.get('model.architecture.dropout_rate', 0.2),
            'batch_size': self.config.get('model.training.batch_size', 32)
        }
