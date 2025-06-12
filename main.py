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

            # Walidacja danych wejściowych
            if not all(key in data_splits for key in
                       ['X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val', 'scalers']):
                raise ValueError("Brakujące klucze w data_splits")

            if not all(isinstance(data_splits[key], np.ndarray) for key in ['X_train', 'y_train', 'X_test', 'y_test']):
                raise ValueError("Dane treningowe i testowe muszą być tablicami numpy")

            # Logowanie informacji o kształcie danych
            self.logger.info(f"Training data shape: {data_splits['X_train'].shape}")
            self.logger.info(f"Validation data shape: {data_splits['X_val'].shape}")
            self.logger.info(f"Test data shape: {data_splits['X_test'].shape}")

            # Step 2: Model training with cross-validation
            self.logger.info("Step 2: Training model...")

            # Sprawdzenie i połączenie zbiorów treningowego i walidacyjnego
            if data_splits['X_val'].size == 0 or data_splits['y_val'].size == 0:
                self.logger.warning("Zbiór walidacyjny jest pusty. Używam tylko danych treningowych...")
                X_train_full = data_splits['X_train']
                y_train_full = data_splits['y_train']
            else:
                # Sprawdzenie zgodności wymiarów
                if data_splits['X_train'].shape[1] != data_splits['X_val'].shape[1]:
                    raise ValueError(f"Niezgodność wymiarów: X_train ({data_splits['X_train'].shape[1]} cech) "
                                     f"vs X_val ({data_splits['X_val'].shape[1]} cech)")

                X_train_full = np.vstack([data_splits['X_train'], data_splits['X_val']])
                y_train_full = np.concatenate([data_splits['y_train'], data_splits['y_val']])

            # Trenowanie modelu
            training_results = self.trainer.cross_validate(X_train_full, y_train_full)

            # Step 3: Final evaluation on test set
            self.logger.info("Step 3: Evaluating on test set...")
            if 'best_model' not in training_results:
                raise ValueError("Brak najlepszego modelu w wynikach treningu")

            test_metrics, test_predictions = self.trainer.evaluate_model(
                training_results['best_model'],
                data_splits['X_test'],
                data_splits['y_test'],
                data_splits['scalers']['scaler_y']
            )

            # Denormalizacja danych testowych
            try:
                y_test_denorm = data_splits['scalers']['scaler_y'].inverse_transform(
                    data_splits['y_test'].reshape(-1, 1)
                ).squeeze()
            except Exception as e:
                self.logger.error(f"Błąd podczas denormalizacji danych: {str(e)}")
                raise

            # Step 4: Generate visualizations
            self.logger.info("Step 4: Generating visualizations...")
            try:
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
            except Exception as e:
                self.logger.error(f"Błąd podczas generowania wizualizacji: {str(e)}")
                # Kontynuujemy wykonanie, ponieważ wizualizacje nie są krytyczne

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
            try:
                self.visualizer.create_results_summary(results)
            except Exception as e:
                self.logger.error(f"Błąd podczas tworzenia podsumowania: {str(e)}")

            self.logger.info("Pipeline completed successfully")
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
            logging.StreamHandler()
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