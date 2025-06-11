"""
Main orchestration module for the enhanced neural network project
"""

import logging
from datetime import datetime
from typing import Dict, Any
import numpy as np

# Import custom modules
from src.config.config_manager import ConfigManager
from src.data.data_loader import DataLoader
from src.models.neural_network import ModelTrainer
from src.models.grid_search import GridSearch
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

        # Run in selected mode based on configuration
        if pipeline_mode == "grid_search":
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
