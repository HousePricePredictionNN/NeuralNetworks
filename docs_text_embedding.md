# Text Embedding and Categorical Feature Handling in Neural Network

## Overview

This project now supports advanced handling of categorical/text features for neural network models. Both low-cardinality and high-cardinality categorical columns are encoded and fed into the model, improving predictive power and flexibility.

## Encoding Strategy

-   **Low-cardinality categoricals** (few unique values, e.g. `city`, `property_type`, `ownership_type`, `voivodeship`):
    -   Encoded using **one-hot encoding** (via pandas `get_dummies`).
-   **High-cardinality categoricals** (many unique values, e.g. `adress`):
    -   Encoded as integer indices and passed to the model as separate columns.
    -   The neural network uses a **learnable embedding layer** (`nn.Embedding`) for each such column.
-   **Binary columns** (e.g. `has_parking`):
    -   Converted to 0/1 if not already numeric.

## Pipeline Changes

-   The data loader (`src/data/data_loader.py`) automatically detects and encodes categorical columns.
-   Embedding metadata is passed to the model, which builds embedding layers for high-cardinality features.
-   The neural network (`src/models/neural_network.py`) concatenates embedding outputs with other features before passing them through the main network.

## How to Configure

-   You can specify which columns to treat as embeddings or one-hot in your config (see `model_config.yaml`):
    ```yaml
    data:
        categorical:
            embedding_columns: ["adress"]
            onehot_columns:
                ["city", "property_type", "ownership_type", "voivodeship"]
    ```
-   The embedding dimension can be set via `model.architecture.embedding_dim` (default: 8).

## Extending/Customizing

-   To add more embedding columns, simply add them to `embedding_columns` in your config.
-   To use a different embedding size, set `model.architecture.embedding_dim`.
-   All other categorical columns not listed are dropped or one-hot encoded as appropriate.

## Example

Suppose your data has:

-   `city` (10 unique values)
-   `adress` (1000+ unique values)

The pipeline will:

-   One-hot encode `city` (10 → 9 columns)
-   Convert `adress` to integer indices and use an embedding layer
-   Concatenate all features for the neural network

## Benefits

-   **Efficient**: Embeddings reduce dimensionality for high-cardinality features
-   **Flexible**: Easily add/remove categorical columns via config
-   **Powerful**: Model can learn relationships between categories

---

For more details, see the code in `src/data/data_loader.py` and `src/models/neural_network.py`.
