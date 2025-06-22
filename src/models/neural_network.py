"""
Enhanced Neural Network Model with improved architecture and training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from pathlib import Path


class NeuralNetwork(nn.Module):
    """
    Enhanced neural network with configurable architecture using Sequential
    Now supports embedding layers for high-cardinality categorical features.
    """

    def __init__(self, input_size: int, config_manager, embedding_info: Optional[Dict] = None):
        super(NeuralNetwork, self).__init__()
        self.config = config_manager
        self.input_size = input_size
        self.embedding_info = embedding_info or {}

        # Get architecture parameters
        hidden_layers = self.config.get(
            "model.architecture.hidden_layers", [128, 64, 32]
        )
        dropout_rate = self.config.get("model.architecture.dropout_rate", 0.2)
        activation = self.config.get("model.architecture.activation", "relu")

        # Get activation function
        if activation == "relu":
            activation_fn = nn.ReLU(True)
        elif activation == "leaky_relu":
            activation_fn = nn.LeakyReLU()
        elif activation == "elu":
            activation_fn = nn.ELU()
        elif activation == "swish":
            activation_fn = nn.SiLU()
        else:
            activation_fn = nn.ReLU(True)

        # Embedding layers for high-cardinality categoricals
        self.embeddings = nn.ModuleDict()
        embedding_output_dim = self.config.get("model.architecture.embedding_dim", 8)
        total_embedding_dim = 0
        
        if self.embedding_info:
            for col, info in self.embedding_info.items():
                num_embeddings = info["num_embeddings"]
                # Use min(50, num_embeddings//2) as default embedding dim if not set
                emb_dim = min(embedding_output_dim, max(2, num_embeddings // 2))
                self.embeddings[col] = nn.Embedding(num_embeddings, emb_dim)
                total_embedding_dim += emb_dim

        # Build Sequential model
        layers = []
        current_size = input_size + total_embedding_dim

        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size

        # Output layer (no activation for regression)
        layers.append(nn.Linear(current_size, 1))

        # Create Sequential model
        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # Check if this is the output layer (last Linear layer)
                if module == list(self.model.modules())[-1]:
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="linear"
                    )
                else:
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
                nn.init.constant_(module.bias, 0)
        
        # Initialize embeddings if they exist
        for emb in self.embeddings.values():
            if hasattr(emb, 'weight') and isinstance(emb.weight, torch.Tensor):
                nn.init.xavier_uniform_(emb.weight)

    def forward(self, x):
        """Forward pass through the network, with embedding support"""
        if self.embeddings:
            # x is expected to be a tensor or numpy array
            # Split x into dense and embedding index parts
            dense_x = x
            emb_list = []
            for col, emb in self.embeddings.items():
                col_idx = self.embedding_info[col]["col_idx"]
                emb_input = x[:, col_idx].long()
                emb_list.append(emb(emb_input))
            if emb_list:
                emb_cat = torch.cat(emb_list, dim=1)
                # Remove embedding index columns from dense_x
                dense_mask = [
                    i
                    for i in range(x.shape[1])
                    if i
                    not in [self.embedding_info[c]["col_idx"] for c in self.embeddings]
                ]
                dense_x = x[:, dense_mask]
                x = torch.cat([dense_x, emb_cat], dim=1)
        return self.model(x)


class ModelTrainer:
    """
    Enhanced model trainer with cross-validation and advanced features
    """

    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def create_model(
        self, input_size: int, embedding_info: Optional[Dict] = None
    ) -> NeuralNetwork:
        """Create model instance with embedding info"""
        model = NeuralNetwork(input_size, self.config, embedding_info)
        return model.to(self.device)

    def train_single_fold(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Train model for a single fold"""

        # Get training parameters
        epochs = self.config.get("model.training.epochs", 1000)
        learning_rate = self.config.get("model.training.learning_rate", 0.001)
        batch_size = self.config.get("model.training.batch_size", 32)

        # Early stopping parameters
        early_stopping_enabled = self.config.get(
            "model.training.early_stopping.enabled", True
        )
        patience = self.config.get("model.training.early_stopping.patience", 50)
        min_delta = self.config.get("model.training.early_stopping.min_delta", 0.001)

        # Setup optimizer and loss
        optimizer_name = self.config.get("model.optimization.optimizer", "adam")
        weight_decay = self.config.get(
            "model.optimization.optimizer_params.weight_decay", 0.0001
        )

        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        loss_fn = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience // 2, factor=0.5
        )

        # Convert to tensors
        X_train_tensor = torch.from_numpy(X_train).float().to(self.device)
        y_train_tensor = torch.from_numpy(y_train).float().to(self.device)
        X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
        y_val_tensor = torch.from_numpy(y_val).float().to(self.device)

        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        self.logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            # Mini-batch training
            n_batches = len(X_train_tensor) // batch_size + (
                1 if len(X_train_tensor) % batch_size != 0 else 0
            )

            for i in range(0, len(X_train_tensor), batch_size):
                batch_end = min(i + batch_size, len(X_train_tensor))
                X_batch = X_train_tensor[i:batch_end]
                y_batch = y_train_tensor[i:batch_end]

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = loss_fn(outputs.squeeze(), y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= n_batches

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = loss_fn(val_outputs.squeeze(), y_val_tensor).item()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if early_stopping_enabled:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()

            # Log progress
            if (epoch + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Epoch {epoch+1:4d}/{epochs}: Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
                )

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        return {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "training_time": training_time,
        }

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, embedding_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation training with embedding info"""
        cv_enabled = self.config.get("model.cross_validation.enabled", True)
        n_folds = self.config.get("model.cross_validation.folds", 5)

        if not cv_enabled:
            # Simple train-validation split
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            model = self.create_model(X.shape[1], embedding_info)
            result = self.train_single_fold(model, X_train, y_train, X_val, y_val)

            return {
                "best_model": result["model"],
                "cv_scores": [result["best_val_loss"]],
                "mean_score": result["best_val_loss"],
                "std_score": 0.0,
                "all_train_losses": [result["train_losses"]],
                "all_val_losses": [result["val_losses"]],
                "training_times": [result["training_time"]],
            }

        # Cross-validation
        self.logger.info(f"Starting {n_folds}-fold cross-validation...")

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = []
        best_model = None
        best_score = float("inf")

        all_train_losses = []
        all_val_losses = []
        training_times = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            self.logger.info(f"Training fold {fold + 1}/{n_folds}")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Create new model for each fold
            model = self.create_model(X.shape[1], embedding_info)

            # Train model
            result = self.train_single_fold(
                model, X_train_fold, y_train_fold, X_val_fold, y_val_fold
            )

            cv_results.append(result["best_val_loss"])
            all_train_losses.append(result["train_losses"])
            all_val_losses.append(result["val_losses"])
            training_times.append(result["training_time"])

            # Keep best model
            if result["best_val_loss"] < best_score:
                best_score = result["best_val_loss"]
                best_model = result["model"]

            self.logger.info(
                f"Fold {fold + 1} completed: Val Loss = {result['best_val_loss']:.6f}"
            )

        mean_score = np.mean(cv_results)
        std_score = np.std(cv_results)

        self.logger.info(f"Cross-validation completed:")
        self.logger.info(f"  Mean CV Score: {mean_score:.6f} (+/- {std_score:.6f})")
        self.logger.info(f"  Best Score: {best_score:.6f}")

        return {
            "best_model": best_model,
            "cv_scores": cv_results,
            "mean_score": mean_score,
            "std_score": std_score,
            "all_train_losses": all_train_losses,
            "all_val_losses": all_val_losses,
            "training_times": training_times,
        }

    def evaluate_model(
        self, model: nn.Module, X: np.ndarray, y: np.ndarray, scaler_y=None
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model performance"""
        model.eval()

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            predictions = model(X_tensor).cpu().numpy().squeeze()

        # Denormalize if scaler provided
        if scaler_y is not None:
            predictions = scaler_y.inverse_transform(
                predictions.reshape(-1, 1)
            ).squeeze()
            y_actual = scaler_y.inverse_transform(y.reshape(-1, 1)).squeeze()
        else:
            y_actual = y

        # Calculate metrics
        mae = mean_absolute_error(y_actual, predictions)
        mse = mean_squared_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, predictions)

        # Calculate percentage errors
        percentage_errors = np.abs((predictions - y_actual) / y_actual) * 100
        mape = np.mean(percentage_errors)

        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}

        return metrics, predictions

    def predict(self, model: nn.Module, X: np.ndarray, scaler_y=None) -> np.ndarray:
        """Make predictions with the model"""
        model.eval()

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            predictions = model(X_tensor).cpu().numpy().squeeze()

        # Denormalize if scaler provided
        if scaler_y is not None:
            predictions = scaler_y.inverse_transform(
                predictions.reshape(-1, 1)
            ).squeeze()

        return predictions

    def save_model(
        self, 
        model: nn.Module, 
        output_dir: Union[str, Path], 
        model_filename: str = "best_model.pth",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save the trained model along with metadata.
        
        Args:
            model: The trained PyTorch model
            output_dir: Directory where to save the model
            model_filename: Name of the model file
            metadata: Additional metadata to save with the model
            
        Returns:
            Path to the saved model file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / model_filename
        
        # Save model state dict with basic config
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': getattr(model, 'input_size', None),
                'embedding_info': getattr(model, 'embedding_info', None)
            },
            'metadata': metadata
        }, model_path)
        
        self.logger.info(f"Model saved to: {model_path}")
        return str(model_path)

    def load_model(
        self, 
        model_path: Union[str, Path]
    ) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
        """
        Load a saved model and its metadata.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Tuple of (loaded model, metadata dict)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config
        model_config = checkpoint.get('model_config', {})
        input_size = model_config.get('input_size')
        embedding_info = model_config.get('embedding_info')
        
        if input_size is None:
            raise ValueError("Model input_size not found in saved model")
        
        # Recreate model
        model = self.create_model(
            input_size=input_size, 
            embedding_info=embedding_info
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get metadata
        metadata = checkpoint.get('metadata')
                
        self.logger.info(f"Model loaded from: {model_path}")
        return model, metadata

    def load_and_predict(
        self, 
        model_path: Union[str, Path], 
        X_new: np.ndarray,
        scaler_y=None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Load a saved model and make predictions on new data.
        
        Args:
            model_path: Path to the saved model file
            X_new: New data to make predictions on
            scaler_y: Target scaler for denormalization (optional)
            
        Returns:
            Tuple of (predictions, metadata dict)
        """
        # Load the model
        model, metadata = self.load_model(model_path)
        
        # Make predictions
        predictions = self.predict(model, X_new, scaler_y)
        
        self.logger.info(f"Made predictions for {len(X_new)} samples using loaded model")
        
        return predictions, metadata
