"""
Simplified visualization module for neural network results
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for better plots
plt.style.use('default')

class ResultsVisualizer:
    """
    Simplified visualization for neural network training and results
    """
    
    def __init__(self, config_manager, output_dir: Path):
        self.config = config_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_training_curves(self, train_losses: List[List[float]], 
                           val_losses: List[List[float]], 
                           cv_scores: Optional[List[float]] = None) -> str:
        """Plot simple training and validation loss curves"""
        
        if not self.config.get('output.plots.loss_curve', True):
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot loss curves - use first fold or average if multiple folds
        if len(train_losses) > 1:  # Multiple folds - plot average
            # Calculate average losses across folds
            max_epochs = max(len(losses) for losses in train_losses)
            
            # Pad sequences and calculate average
            padded_train = []
            padded_val = []
            for train_loss, val_loss in zip(train_losses, val_losses):
                padded_train.append(np.pad(train_loss, (0, max_epochs - len(train_loss)), 
                                         constant_values=np.nan))
                padded_val.append(np.pad(val_loss, (0, max_epochs - len(val_loss)), 
                                       constant_values=np.nan))
            
            avg_train = np.nanmean(padded_train, axis=0)
            avg_val = np.nanmean(padded_val, axis=0)
            
            # Remove NaN values for plotting
            valid_epochs_train = np.where(~np.isnan(avg_train))[0]
            valid_epochs_val = np.where(~np.isnan(avg_val))[0]
            
            if len(valid_epochs_train) > 0:
                ax.plot(valid_epochs_train + 1, avg_train[valid_epochs_train], 
                       'b-', label='Training Loss', linewidth=2)
            if len(valid_epochs_val) > 0:
                ax.plot(valid_epochs_val + 1, avg_val[valid_epochs_val], 
                       'r-', label='Validation Loss', linewidth=2)
        else:  # Single fold
            epochs = range(1, len(train_losses[0]) + 1)
            ax.plot(epochs, train_losses[0], 'b-', label='Training Loss', linewidth=2)
            ax.plot(epochs, val_losses[0], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        loss_curve_path = self.output_dir / 'loss_curve.png'
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Loss curve saved to {loss_curve_path}")
        return str(loss_curve_path)
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 dataset_name: str = "Test") -> str:
        """Plot simple predictions vs actual values"""
        
        if not self.config.get('output.plots.predictions_vs_actual', True):
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Prediction')
        
        # Calculate and display metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Add metrics text
        metrics_text = f'R² = {r2:.3f}\nRMSE = {rmse:.0f}\nMAE = {mae:.0f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title(f'Predictions vs Actual Values - {dataset_name} Set')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Make plot square
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save plot
        predictions_path = self.output_dir / 'predictions_vs_actual.png'
        plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Predictions plot saved to {predictions_path}")
        return str(predictions_path)
    
    def create_results_summary(self, results):
        """Create a simple but comprehensive text summary of training results"""
        summary_path = self.output_dir / 'results_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("NEURAL NETWORK TRAINING RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic info
            f.write("PROJECT INFORMATION:\n")
            f.write(f"  Project: {getattr(self.config, 'get', lambda x, default: default)('project.name', 'House Price Prediction')}\n")
            f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Execution Time: {results.get('execution_time', 0):.2f} seconds\n")
            f.write(f"  Output Directory: {self.output_dir}\n\n")
            
            # Data summary
            f.write("DATA SUMMARY:\n")
            f.write(f"  Original Shape: {results.get('original_shape', 'N/A')}\n")
            f.write(f"  Features Used: {results.get('n_features', 'N/A')}\n")
            f.write(f"  Train/Val/Test Split: {results.get('n_train', 0):,} / {results.get('n_val', 0):,} / {results.get('n_test', 0):,}\n\n")
            
            # Model info
            f.write("MODEL CONFIGURATION:\n")
            f.write(f"  Architecture: {getattr(self.config, 'get', lambda x, default: default)('model.architecture.hidden_layers', 'N/A')}\n")
            f.write(f"  Learning Rate: {getattr(self.config, 'get', lambda x, default: default)('model.training.learning_rate', 'N/A')}\n")
            f.write(f"  Batch Size: {getattr(self.config, 'get', lambda x, default: default)('model.training.batch_size', 'N/A')}\n")
            f.write(f"  Max Epochs: {getattr(self.config, 'get', lambda x, default: default)('model.training.epochs', 'N/A')}\n\n")
            
            # Training results
            training = results.get('training_results', {})
            f.write("TRAINING RESULTS:\n")
            f.write(f"  Cross-Validation: {getattr(self.config, 'get', lambda x, default: default)('model.cross_validation.enabled', False)}\n")
            
            if training.get('cv_scores') and len(training['cv_scores']) > 1:
                f.write(f"  CV Score: {training.get('mean_score', 'N/A'):.6f} (+/- {training.get('std_score', 'N/A'):.6f})\n")
            
            best_val_loss = training.get('best_val_loss', 'N/A')
            if isinstance(best_val_loss, (int, float)):
                f.write(f"  Best Validation Loss: {best_val_loss:.6f}\n")
            
            training_times = training.get('training_times', [])
            if training_times:
                f.write(f"  Training Time: {sum(training_times):.2f} seconds\n")
            f.write("\n")
            
            # Test performance
            if 'test_metrics' in results:
                metrics = results['test_metrics']
                f.write("TEST SET PERFORMANCE:\n")
                f.write(f"  R2 Score: {metrics.get('r2', 'N/A'):.4f}\n")
                f.write(f"  MAE: {metrics.get('mae', 'N/A'):,.0f}\n")
                f.write(f"  RMSE: {metrics.get('rmse', 'N/A'):,.0f}\n")
                f.write(f"  MAPE: {metrics.get('mape', 'N/A'):.1f}%\n\n")
                
                # Simple performance assessment
                if 'r2' in metrics:
                    r2 = metrics['r2']
                    if r2 >= 0.8:
                        assessment = "Good"
                    elif r2 >= 0.6:
                        assessment = "Fair"
                    else:
                        assessment = "Poor"
                    f.write(f"  Performance: {assessment} (explains {r2*100:.1f}% of variance)\n\n")
            
            # Files generated
            f.write("FILES GENERATED:\n")
            for file_path in sorted(self.output_dir.glob('*')):
                f.write(f"  {file_path.name}\n")
            
            f.write("\n" + "=" * 60)
        
        self.logger.info(f"Results summary saved to {summary_path}")