"""
Grid Search Module for Neural Network Project
Handles hyperparameter optimization through grid search
"""

import logging
import itertools
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
import copy
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Import custom modules
from src.models.neural_network import ModelTrainer


class GridSearch:
    """
    Grid Search for hyperparameter optimization of the neural network models
    """
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.best_params = None
        self.best_score = float('inf')  # For minimization metrics like MSE
        self.best_model = None
        self.trainer = ModelTrainer(config_manager)
        
    def define_param_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for search based on configuration
        
        Returns:
            Dict containing parameter grid
        """
        # Check for grid in configuration
        param_grid = self.config.get('model.grid_search.param_grid', None)
        
        if not param_grid:
            self.logger.warning("No parameter grid defined in configuration")
            return {}
        
        # Log the parameter grid
        self.logger.info(f"Grid search parameter grid defined with {self._count_combinations(param_grid)} combinations")
        for param, values in param_grid.items():
            self.logger.debug(f"Parameter '{param}' values: {values}")
            
        return param_grid

    def _count_combinations(self, param_grid: Dict[str, List[Any]]) -> int:
        """Count total number of parameter combinations"""
        return int(np.prod([len(values) for values in param_grid.values()]))
    
    def _apply_params(self, params: Dict[str, Any]):
        """Apply parameters to the configuration manager"""
        # Check if params is None or empty
        if params is None:
            self.logger.warning("No parameters to apply - params is None")
            return
        
        if not isinstance(params, dict):
            self.logger.warning(f"Invalid params type: {type(params)}. Expected dict.")
            return
        
        # Set model architecture parameters
        if "hidden_layers" in params:
            self.config.set("model.architecture.hidden_layers", params["hidden_layers"])
        if "dropout_rate" in params:
            self.config.set("model.architecture.dropout_rate", params["dropout_rate"])
        if "activation" in params:
            self.config.set("model.architecture.activation", params["activation"])
            
        # Set training parameters
        if "learning_rate" in params:
            self.config.set("model.training.learning_rate", params["learning_rate"])
        if "batch_size" in params:
            self.config.set("model.training.batch_size", params["batch_size"])
        if "optimizer" in params:
            self.config.set("model.optimization.optimizer", params["optimizer"])
            
        # Set regularization parameters
        if "weight_decay" in params:
            self.config.set("model.optimization.optimizer_params.weight_decay", params["weight_decay"])
            
        # Set scaler type
        if "scaler_type" in params:
            self.config.set("data.preprocessing.scaler_type", params["scaler_type"])
    
    def _get_scaler(self, scaler_type: str):
        """Get the appropriate scaler based on type"""
        if scaler_type == "robust":
            return RobustScaler()
        elif scaler_type == "minmax":
            return MinMaxScaler()
        else:  # default to standard
            return StandardScaler()
        
    def _evaluate_params(self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float], Any]:
        """
        Evaluate a single set of parameters
        
        Args:
            params: Dictionary of parameters to test
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Tuple of (params, metrics)
        """
        start_time = time.time()
        self.logger.info(f"Evaluating parameters: {params}")
        
        # Apply parameters to config
        self._apply_params(params)
        
        # Check if we need to rescale based on param grid
        if "scaler_type" in params:
            # Create new scalers based on parameter
            x_scaler = self._get_scaler(params["scaler_type"])
            y_scaler = self._get_scaler(params["scaler_type"])
            
            # Scale the data
            X_train_scaled = x_scaler.fit_transform(X_train)
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            X_val_scaled = x_scaler.transform(X_val)
            y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
        else:
            # Use data as is
            X_train_scaled = X_train
            y_train_scaled = y_train
            X_val_scaled = X_val
            y_val_scaled = y_val
            y_scaler = None        # Create and train model
        model = self.trainer.create_model(X_train.shape[1])

        training_results = self.trainer.train_single_fold(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled
        )
        
        # Get metrics on validation set
        raw_metrics, _ = self.trainer.evaluate_model(
            training_results['model'], 
            X_val_scaled,
            y_val_scaled,
            y_scaler
        )
        
        # Store the trained model for potential best model tracking
        trained_model = training_results['model']
        
        # Map metrics to expected keys for grid search
        metrics = {
            "val_mse": raw_metrics.get("mse", 0.0) if isinstance(raw_metrics, dict) else raw_metrics[0],
            "val_mae": raw_metrics.get("mae", 0.0) if isinstance(raw_metrics, dict) else raw_metrics[1], 
            "val_r2": raw_metrics.get("r2", 0.0) if isinstance(raw_metrics, dict) else raw_metrics[2],
            "val_rmse": raw_metrics.get("rmse", 0.0) if isinstance(raw_metrics, dict) else raw_metrics[3],
            "val_mape": raw_metrics.get("mape", 0.0) if isinstance(raw_metrics, dict) else raw_metrics[4]
        }
          # Record execution time
        execution_time = time.time() - start_time
        metrics["execution_time"] = float(execution_time)
        
        # Ensure all metric values are floats
        float_metrics = {}
        for key, value in metrics.items():
            try:
                float_metrics[key] = float(value)
            except (ValueError, TypeError):
                self.logger.warning(f"Could not convert metric {key} to float: {value}")
                float_metrics[key] = 0.0
        
        # Return results including the trained model
        return params, float_metrics, trained_model
    
    def search(self, X_train: np.ndarray, y_train: np.ndarray, 
               X_val: np.ndarray, y_val: np.ndarray,
               n_folds: int = 3, 
               optimization_metric: str = "val_mse") -> Dict[str, Any]:
        """
        Perform grid search to find optimal hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            n_folds: Number of cross-validation folds
            optimization_metric: Metric to optimize ('val_mse', 'val_mae', 'val_r2')
            
        Returns:
            Dictionary containing best parameters and results
        """
        # Get parameter grid from configuration or default
        param_grid = self.define_param_grid()
            
        # Calculate total combinations
        total_combinations = self._count_combinations(param_grid)
        self.logger.info(f"Starting grid search with {total_combinations} combinations")
        
        # Initialize results storage
        self.results = []
        
        # Generate all parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        # Main grid search loop
        progress_bar = tqdm(total=total_combinations, desc="Grid Search Progress")
        
        for combination in itertools.product(*values):
            # Create parameter dictionary
            params = {keys[i]: combination[i] for i in range(len(keys))}
              # Log progress
            self.logger.info(f"Evaluating combination {progress_bar.n+1}/{total_combinations} ({(progress_bar.n+1)/total_combinations*100:.1f}%)")
            
            try:
                # Evaluate parameters
                params, metrics, trained_model = self._evaluate_params(params, X_train, y_train, X_val, y_val)
                
                # Validate metrics
                if not isinstance(metrics, dict):
                    self.logger.error(f"Invalid metrics type: {type(metrics)}")
                    progress_bar.update(1)
                    continue
                
                # Check for required metrics
                required_metrics = ["val_mse", "val_mae", "val_r2"]
                missing_metrics = [m for m in required_metrics if m not in metrics]
                if missing_metrics:
                    self.logger.error(f"Missing metrics: {missing_metrics}")
                    progress_bar.update(1)
                    continue
                
                # Determine score based on optimization metric
                if optimization_metric == "val_r2":
                    score = -metrics["val_r2"]  # Negate for maximization
                elif optimization_metric == "val_mae":
                    score = metrics["val_mae"]
                else:  # Default to MSE
                    score = metrics["val_mse"]
                
                # Validate score
                if not isinstance(score, (int, float)) or np.isnan(score) or np.isinf(score):
                    self.logger.error(f"Invalid score: {score} for params: {params}")
                    progress_bar.update(1)
                    continue
                  # Check if this is the best score
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = copy.deepcopy(params)
                    self.best_model = trained_model  # Store the best model
                    
                # Store results
                result = {**params, **metrics}
                self.results.append(result)
                
                # Log results  
                self.logger.info(f"Parameters: {params}")
                self.logger.info(f"Metrics: MSE={metrics['val_mse']:.4f}, MAE={metrics['val_mae']:.4f}, R²={metrics['val_r2']:.4f}")
                self.logger.info(f"Best score so far: {self.best_score:.4f} (lower is better)")
                
                # Update progress bar
                progress_bar.set_postfix(best_score=f"{self.best_score:.4f}")
                progress_bar.update(1)
                
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                progress_bar.update(1)

        progress_bar.close()
        
        # Check if we found any valid results
        if not self.results:
            self.logger.error("No parameter combinations were successfully evaluated!")
            raise RuntimeError("Grid search failed - no valid parameter combinations found")
        
        self.logger.info(f"Grid search completed successfully!")
        self.logger.info(f"Total valid results: {len(self.results)}")
        self.logger.info(f"Best score: {self.best_score:.6f}")
        self.logger.info(f"Best parameters: {self.best_params}")
          # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Apply best parameters to config
        if self.best_params:
            self._apply_params(self.best_params)
          # Return best parameters and additional information
        return {
            "best_params": self.best_params,
            "best_model": self.best_model,
            "best_score": -self.best_score if optimization_metric == "val_r2" else self.best_score,
            "optimization_metric": optimization_metric,
            "results": results_df,
            "total_combinations": total_combinations,
            "completed_combinations": len(self.results)
        }
        
    def save_results(self, output_dir: Path) -> str:
        """
        Save grid search results to CSV
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to saved results file
        """
        if not self.results:
            self.logger.warning("No grid search results to save")
            return ""
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Sort by the optimization metric
        if "val_mse" in results_df.columns:
            results_df = results_df.sort_values(by="val_mse")
          # Save to CSV
        output_path = output_dir / "grid_search_results.csv"
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"Grid search results saved to {output_path}")
        
        return str(output_path)
    
    def visualize_results(self, output_dir: Path, visualizer=None) -> Dict[str, str]:
        """
        Create simple visualizations of grid search results using the plotting module
        
        Args:
            output_dir: Directory to save visualizations
            visualizer: ResultsVisualizer instance (optional)
            
        Returns:
            Dictionary of paths to saved visualizations
        """
        if not self.results:
            self.logger.warning("No grid search results to visualize")
            return {}
        
        saved_paths = {}
        
        try:
            results_df = pd.DataFrame(self.results)
            
            if visualizer:
                # Use the plotting module for visualization
                plot_paths = visualizer.plot_grid_search_results(results_df)
                for i, path in enumerate(plot_paths):
                    saved_paths[f"plot_{i+1}"] = path
            else:
                self.logger.info("No visualizer provided, skipping visualization")
                
        except Exception as e:
            self.logger.error(f"Error creating grid search visualizations: {str(e)}")
        
        return saved_paths
