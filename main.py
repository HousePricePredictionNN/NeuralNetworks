"""
Main orchestration module for the enhanced neural network project
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Import custom modules
from src.config.config_manager import ConfigManager
from src.data.data_loader import DataLoader
from src.models.neural_network import ModelTrainer
from src.optimization.hyperparameter_tuning import HyperparameterOptimizer
from src.visualization.plotting import ResultsVisualizer

class NeuralNetworkPipeline:
    """
    Main pipeline for house price prediction using neural networks
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        # Initialize components
        self.config = ConfigManager(config_path)
        self.data_loader = DataLoader(self.config)
        self.trainer = ModelTrainer(self.config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.output_dir = self.config.get_output_dir() / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = ResultsVisualizer(self.config, self.output_dir)
        
        self.logger.info(f"Pipeline initialized. Output directory: {self.output_dir}")
        
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete neural network pipeline"""
        
        self.logger.info("Starting complete neural network pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Data preparation
            self.logger.info("Step 1: Preparing data...")
            data_splits = self.data_loader.prepare_data_pipeline()
            
            # Step 2: Hyperparameter optimization (if enabled)
            if self.config.get('model.grid_search.enabled', False):
                self.logger.info("Step 2: Hyperparameter optimization...")
                optimizer = HyperparameterOptimizer(self.config)
                best_params = optimizer.optimize(data_splits['X_train'], data_splits['y_train'])
                self.logger.info(f"Best parameters found: {best_params}")
            else:
                self.logger.info("Step 2: Skipping hyperparameter optimization...")
            
            # Step 3: Model training with cross-validation
            self.logger.info("Step 3: Training model...")
            
            # Combine train and validation for cross-validation
            X_train_full = np.vstack([data_splits['X_train'], data_splits['X_val']])
            y_train_full = np.concatenate([data_splits['y_train'], data_splits['y_val']])
            
            training_results = self.trainer.cross_validate(X_train_full, y_train_full)
            
            # Step 4: Final evaluation on test set
            self.logger.info("Step 4: Evaluating on test set...")
            test_metrics, test_predictions = self.trainer.evaluate_model(
                training_results['best_model'], 
                data_splits['X_test'], 
                data_splits['y_test'],
                data_splits['scalers']['scaler_y']
            )
            
            # Denormalize test targets for visualization
            y_test_denorm = data_splits['scalers']['scaler_y'].inverse_transform(
                data_splits['y_test'].reshape(-1, 1)
            ).squeeze()
            
            # Step 5: Generate visualizations
            self.logger.info("Step 5: Generating visualizations...")
            
            # Training curves
            self.visualizer.plot_training_curves(
                training_results['all_train_losses'],
                training_results['all_val_losses'], 
                training_results['cv_scores']
            )
            
            # Predictions vs actual
            self.visualizer.plot_predictions_vs_actual(
                y_test_denorm, test_predictions, "Test"
            )
            
            # Create results summary
            results = {
                'original_shape': data_splits.get('original_shape', 'N/A'),
                'n_features': data_splits['X_train'].shape[1],
                'n_train': len(data_splits['X_train']),
                'n_val': len(data_splits['X_val']),
                'n_test': len(data_splits['X_test']),
                'training_results': training_results,
                'test_metrics': test_metrics,
                'output_directory': str(self.output_dir),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Create text summary
            self.visualizer.create_results_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the neural network pipeline"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('neural_network.log')
        ]
    )
    
    try:
        # Initialize and run pipeline
        pipeline = NeuralNetworkPipeline()
        results = pipeline.run_complete_pipeline()
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {results['output_directory']}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()