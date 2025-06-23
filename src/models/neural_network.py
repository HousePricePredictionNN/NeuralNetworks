"""
Enhanced Neural Network Model with improved architecture and training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
import copy

class NeuralNetwork(nn.Module):
    """
    Enhanced neural network with configurable architecture using Sequential
    Now supports embedding layers for high-cardinality categorical features.
    """

    def __init__(self, input_size: int, config_manager, embedding_info: dict = None):
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
        for col, info in (self.embedding_info or {}).items():
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
        for emb in self.embeddings.values():
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
        self, input_size: int, embedding_info: dict = None
    ) -> NeuralNetwork:
        """Create model instance with embedding info"""
        model = NeuralNetwork(input_size, self.config, embedding_info)
        return model.to(self.device)

    def predict_future_prices(self, model, last_sequence, num_years=5):
        """
        Generuje predykcje na kolejne lata na podstawie ostatniej znanej sekwencji

        Args:
            model: Wytrenowany model LSTM
            last_sequence: Ostatnia znana sekwencja danych
            num_years: Liczba lat do przewidzenia w przyszłość

        Returns:
            DataFrame z predykcjami
        """
        model.eval()
        predictions = []
        current_sequence = last_sequence.copy()

        with torch.no_grad():
            for _ in range(num_years):
                # Przygotuj sekwencję do predykcji
                sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)

                # Generuj predykcję
                pred = model(sequence_tensor)
                pred_value = pred.cpu().numpy()[0][0]
                predictions.append(pred_value)

                # Aktualizuj sekwencję dla następnej predykcji
                # Usuń najstarszy punkt i dodaj nową predykcję
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, -1] = pred_value  # Zakładamy, że cena jest ostatnią cechą

        return predictions

    def train_single_fold(self, X_train, y_train, X_val, y_val, embedding_info=None) -> Dict[str, Any]:
        """Train model on a single fold"""
        self.logger.info("Starting training for %d epochs...", self.config.get("model.training.epochs"))

        # Initialize model
        input_size = X_train.shape[1]
        model = self.create_model(input_size, embedding_info)

        # Training parameters
        epochs = self.config.get("model.training.epochs", 1000)
        batch_size = self.config.get("model.training.batch_size", 32)
        learning_rate = self.config.get("model.training.learning_rate", 0.001)

        # Early stopping parameters
        early_stopping = self.config.get("model.training.early_stopping", {})
        patience = early_stopping.get("patience", 50)
        min_delta = early_stopping.get("min_delta", 0.001)

        # Initialize optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get("model.optimization.optimizer_params.weight_decay", 0.0001)
        )

        # Track best model and metrics
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []

        # Przygotuj progress bar
        pbar = tqdm(range(epochs), desc="Training Progress")

        try:
            for epoch in pbar:
                model.train()
                total_loss = 0
                num_batches = 0

                # Training loop
                for i in range(0, len(X_train), batch_size):
                    batch_X = torch.FloatTensor(X_train[i:i + batch_size]).to(self.device)
                    batch_y = torch.FloatTensor(y_train[i:i + batch_size]).to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = F.mse_loss(outputs, batch_y.view(-1, 1))
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                avg_train_loss = total_loss / num_batches
                train_losses.append(avg_train_loss)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_X = torch.FloatTensor(X_val).to(self.device)
                    val_y = torch.FloatTensor(y_val).to(self.device)
                    val_outputs = model(val_X)
                    val_loss = F.mse_loss(val_outputs, val_y.view(-1, 1)).item()
                    val_losses.append(val_loss)

                # Aktualizuj pasek postępu
                pbar.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })

                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        finally:
            pbar.close()

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }

    def train_lstm(self, sequences, targets, val_sequences=None, val_targets=None):
        """
        Trenuje model LSTM na danych sekwencyjnych
        """
        input_size = sequences.shape[2]  # liczba cech
        hidden_size = self.config.get('lstm_hidden_size', 64)
        num_layers = self.config.get('lstm_num_layers', 2)
        output_size = 1

        model = LSTMNetwork(input_size, hidden_size, num_layers, output_size)
        model = model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.001))

        num_epochs = self.config.get('num_epochs', 100)
        batch_size = self.config.get('batch_size', 32)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for i in range(0, len(sequences), batch_size):
                batch_sequences = torch.FloatTensor(sequences[i:i + batch_size]).to(self.device)
                batch_targets = torch.FloatTensor(targets[i:i + batch_size]).to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs.squeeze(), batch_targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(sequences) // batch_size)
            self.logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

            if val_sequences is not None and val_targets is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(torch.FloatTensor(val_sequences).to(self.device))
                    val_loss = criterion(val_outputs.squeeze(),
                                         torch.FloatTensor(val_targets).to(self.device))
                    self.logger.info(f'Validation Loss: {val_loss:.4f}')

        return model

    def cross_validate(self, X, y, n_splits=5, embedding_info=None) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""
        self.logger.info(f"Starting {n_splits}-fold cross-validation...")

        # Initialize lists to store results
        all_train_losses = []
        all_val_losses = []
        cv_scores = []
        best_val_loss = float('inf')
        best_model = None

        # Create KFold object
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Progress bar dla foldów
        fold_pbar = tqdm(enumerate(kf.split(X), 1), total=n_splits, desc="Cross-validation Progress")

        for fold, (train_idx, val_idx) in fold_pbar:
            fold_pbar.set_description(f"Training fold {fold}/{n_splits}")

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train on this fold
            fold_results = self.train_single_fold(X_train, y_train, X_val, y_val, embedding_info)

            # Store results
            all_train_losses.append(fold_results["train_losses"])
            all_val_losses.append(fold_results["val_losses"])
            cv_scores.append(fold_results["best_val_loss"])

            # Update best model if this fold performed better
            if fold_results["best_val_loss"] < best_val_loss:
                best_val_loss = fold_results["best_val_loss"]
                best_model = fold_results["model"]

            # Aktualizuj progress bar
            fold_pbar.set_postfix({
                'best_val_loss': f'{best_val_loss:.4f}',
                'current_fold_loss': f'{fold_results["best_val_loss"]:.4f}'
            })

        fold_pbar.close()

        # Calculate cross-validation metrics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        self.logger.info(f"Cross-validation complete. Mean MSE: {cv_mean:.4f} (±{cv_std:.4f})")

        return {
            "best_model": best_model,
            "all_train_losses": all_train_losses,
            "all_val_losses": all_val_losses,
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std
        }

    def evaluate_model(
        self, model: nn.Module, X: np.ndarray, y: np.ndarray, scaler_y=None
    ) -> Dict[str, float]:
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


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # Bierzemy tylko ostatni output z sekwencji
        out = self.fc(out[:, -1, :])
        return out