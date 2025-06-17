"""
Main orchestration module for the enhanced neural network project
"""

import logging
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import numpy as np

# Import custom modules
from src.config.config_manager import ConfigManager
from src.data.data_loader import DataLoader
from src.models.neural_network import ModelTrainer
from src.models.grid_search import GridSearch
from src.visualization.plotting import ResultsVisualizer


def save_split_datasets_raw(data_loader: DataLoader, output_base_dir: str = "data/splits"):
    """
    Save training, validation, and test datasets as 3 separate CSV files with RAW data
    (before normalization but after preprocessing like categorical encoding).
    Each file contains both features and target values combined.
    Only saves if the files do not already exist.
    
    Args:
        data_loader: DataLoader instance to access data preparation methods
        output_base_dir: Base directory where split files will be created
    """
    logger = logging.getLogger(__name__)
    
    # Create base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    logger.info("Loading and preprocessing raw data for splitting...")
    
    # Load and preprocess data (but don't normalize)
    raw_data = data_loader.load_data()
    processed_data, embedding_info = data_loader.preprocess_data(raw_data)
    
    # Split features and target
    X, y = data_loader.split_features_target(processed_data)
    
    # Create train/val/test splits (this gives us raw preprocessed data)
    raw_splits = data_loader.create_data_splits(X, y)
    
    # Get feature names
    feature_names = data_loader.feature_names
    
    # Define the split names and corresponding data
    splits_info = {
        'training': {
            'X': raw_splits['X_train'],
            'y': raw_splits['y_train']
        },
        'validation': {
            'X': raw_splits['X_val'],
            'y': raw_splits['y_val']
        },
        'test': {
            'X': raw_splits['X_test'],
            'y': raw_splits['y_test']
        }
    }
    
    for split_name, split_data in splits_info.items():
        # Define file path - single file per split
        dataset_file = os.path.join(output_base_dir, f'{split_name}_dataset.csv')
        
        # Check if file already exists
        if os.path.exists(dataset_file):
            logger.info(f"Dataset file for '{split_name}' already exists. Skipping...")
            continue
        
        # Convert to DataFrames and combine
        try:
            # Handle features (X data) with original column names
            if hasattr(split_data['X'], 'columns'):
                # If it's already a DataFrame
                features_df = split_data['X'].copy()
            else:
                # If it's a pandas Series or numpy array, create DataFrame with original column names
                if feature_names is not None:
                    features_df = pd.DataFrame(split_data['X'], columns=feature_names)
                else:
                    features_df = pd.DataFrame(split_data['X'])
            
            # Handle target (y data) - should be pandas Series
            if hasattr(split_data['y'], 'values'):
                # If it's a pandas Series
                target_values = split_data['y'].values
            else:
                # If it's a numpy array
                target_values = split_data['y']
            
            # Add target column to features DataFrame
            features_df['price'] = target_values
            
            # Save the combined dataset
            features_df.to_csv(dataset_file, index=False)
            
            logger.info(f"Saved {split_name} dataset (RAW - before normalization):")
            logger.info(f"  File: {dataset_file}")
            logger.info(f"  Shape: {features_df.shape} (features + target)")
            logger.info(f"  Samples: {len(features_df)}")
            if feature_names:
                logger.info(f"  Column names preserved: {len(feature_names)} features + price")
            
        except Exception as e:
            logger.error(f"Error saving {split_name} dataset: {str(e)}")
            
    logger.info(f"RAW dataset splitting completed. 3 files saved in: {output_base_dir}")
    logger.info("These datasets contain the same splits that will be used for training, but with raw values (before normalization)")


class NeuralNetworkPipeline:
    """
    Main pipeline for house price prediction using neural networks
    """

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        # Initialize components
        self.config = ConfigManager(config_path)
        self.data_loader = DataLoader(self.config)
        self.trainer = ModelTrainer(self.config)
        self.grid_search = GridSearch(self.config)

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
            embedding_info = data_splits.get("embedding_info", None)
            # Step 2: Model training with cross-validation

            # Visualize missing data
            raw_data = self.data_loader.load_data()
            self.visualizer.plot_missing_data(raw_data)

            # Step 2: Model training with cross-validation
            self.logger.info("Step 2: Training model...")

            # Combine train and validation for cross-validation
            X_train_full = np.vstack([data_splits["X_train"], data_splits["X_val"]])
            y_train_full = np.concatenate(
                [data_splits["y_train"], data_splits["y_val"]]
            )
  
            training_results = self.trainer.cross_validate(
                X_train_full, y_train_full, embedding_info=embedding_info
            )

            # Step 3: Final evaluation on test set
            self.logger.info("Step 3: Evaluating on test set...")
            test_metrics, test_predictions = self.trainer.evaluate_model(
                training_results["best_model"],
                data_splits["X_test"],
                data_splits["y_test"],
                data_splits["scalers"]["scaler_y"],
            )

            # Denormalize test targets for visualization
            y_test_denorm = (
                data_splits["scalers"]["scaler_y"]
                .inverse_transform(data_splits["y_test"].reshape(-1, 1))
                .squeeze()
            )
            # Step 4: Generate visualizations
            self.logger.info("Step 4: Generating visualizations...")

            # Training curves
            self.visualizer.plot_training_curves(
                training_results["all_train_losses"],
                training_results["all_val_losses"],
                training_results["cv_scores"],
            )

            # Predictions vs actual
            self.visualizer.plot_predictions_vs_actual(
                y_test_denorm, test_predictions, "Test"
            )

            # Create results summary
            results = {
                "original_shape": data_splits.get("original_shape", "N/A"),
                "n_features": data_splits["X_train"].shape[1],
                "n_train": len(data_splits["X_train"]),
                "n_val": len(data_splits["X_val"]),
                "n_test": len(data_splits["X_test"]),
                "training_results": training_results,
                "test_metrics": test_metrics,
                "output_directory": str(self.output_dir),
                "execution_time": (datetime.now() - start_time).total_seconds(),
            }
            # Create text summary
            self.visualizer.create_results_summary(results)

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def run_grid_search(self) -> Dict[str, Any]:
        self.logger.info("Starting grid search for optimal hyperparameters...")
        start_time = datetime.now()

        # Prepare data
        data_splits = self.data_loader.prepare_data_pipeline()

        # Visualize missing data
        raw_data = self.data_loader.load_data()
        self.visualizer.plot_missing_data(raw_data)

        # Run grid search
        grid_search_results = self.grid_search.search(
            data_splits["X_train"],
            data_splits["y_train"],
            data_splits["X_val"],
            data_splits["y_val"],
            optimization_metric="val_mse",
        )
        # Save and visualize results
        self.grid_search.save_results(self.output_dir)
        self.grid_search.visualize_results(self.output_dir, self.visualizer)

        # Create consolidated results
        results = {
            "original_shape": data_splits.get("original_shape", "N/A"),
            "best_params": grid_search_results["best_params"],
            "best_score": grid_search_results["best_score"],
            "total_combinations": grid_search_results["total_combinations"],
            "completed_combinations": grid_search_results["completed_combinations"],
            "output_directory": str(self.output_dir),
            "execution_time": (datetime.now() - start_time).total_seconds(),
        }

        self.logger.info(
            f"Grid search completed in {results['execution_time']:.2f} seconds"
        )
        self.logger.info(f"Best parameters: {results['best_params']}")
        self.logger.info(f"Best score: {results['best_score']}")

        return results

    def prepare_and_save_datasets_only(self) -> Dict[str, Any]:
        """
        Prepare data and save split datasets only, without any training.
        Returns information about the saved datasets.
        """
        self.logger.info("Preparing and saving split datasets...")
        
        try:
            # Use DataLoader's new method to save datasets
            results = self.data_loader.save_split_datasets_raw()
            
            self.logger.info("Dataset preparation completed successfully!")
            self.logger.info(f"3 RAW dataset files saved: training_dataset.csv, validation_dataset.csv, test_dataset.csv")
            self.logger.info(f"Train samples: {results['n_train']}")
            self.logger.info(f"Validation samples: {results['n_val']}")
            self.logger.info(f"Test samples: {results['n_test']}")
            self.logger.info(f"Features: {results['n_features']} + 1 target column")
            self.logger.info(f"Random state: {results['random_state']} (ensures consistent splits)")
            self.logger.info(f"Data flow: {results['original_shape']} -> {results['processed_shape']} -> splits")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {str(e)}")
            raise


def main():
    """Main function to run the neural network pipeline"""

    # Config path - could be set as an environment variable or hardcoded
    config_path = "configs/model_config.yaml"

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    try: 
        # Initialize pipeline
        pipeline = NeuralNetworkPipeline(config_path=config_path)

        # Get pipeline mode from config
        pipeline_mode = pipeline.config.get("pipeline.mode", "train")      
        if pipeline_mode == "save_datasets":
            logging.info("Running dataset preparation and saving mode")
            results = pipeline.prepare_and_save_datasets_only()
            print("\nDataset preparation completed successfully!")
            print(f"3 dataset files saved in: {results['datasets_saved_to']}")
            print(f"  - training_dataset.csv ({results['n_train']} samples)")
            print(f"  - validation_dataset.csv ({results['n_val']} samples)")
            print(f"  - test_dataset.csv ({results['n_test']} samples)")
            print(f"Each file contains {results['n_features']} features + 1 target column")
            print(f"Random state: {results['random_state']} (ensures consistent splits)")
        elif pipeline_mode == "grid_search":
            logging.info("Running grid search mode")
            results = pipeline.run_grid_search()
            print("\nGrid search completed successfully!")
            print(f"Results saved to: {results['output_directory']}")
            print(f"Best parameters: {results['best_params']}")
        else:
            logging.info("Running standard training mode")
            results = pipeline.run_complete_pipeline()
            print("\nPipeline completed successfully!")
            print(f"Results saved to: {results['output_directory']}")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
