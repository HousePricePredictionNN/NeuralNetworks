# House Price Prediction with Neural Networks

A simple neural network system for predicting house prices. Built with Python and PyTorch, it handles everything from data loading to generating prediction plots.

## What it does

- Loads house price data and splits it properly for training
- Trains a neural network with cross-validation
- Shows you how well the model performs
- Creates simple plots to visualize results

## Quick Start

1. Install the required packages:

```bash
pip install torch numpy pandas scikit-learn matplotlib pyyaml
```

2. Run the system:

```bash
python main.py
```

That's it! The system will train a model and save results in the `data/output/` folder.

## Project Structure

```
NeuralNetworks/
├── configs/model_config.yaml    # Settings and parameters
├── src/                         # Main code
├── data/input/data.csv         # Your house price data
└── data/output/                # Results go here
```

## Configuration

You can adjust settings in `configs/model_config.yaml`:

```yaml
# How much data to use
data:
  loading:
    total_rows: 15000
    train_ratio: 0.7

# Neural network settings
model:
  training:
    epochs: 500
    batch_size: 32
    learning_rate: 0.001
  architecture:
    hidden_layers: [128, 64, 32, 16]
```

## What you get

After training, you'll find these files in a timestamped folder:

- `loss_curve.png` - Shows how training progressed
- `predictions_vs_actual.png` - Scatter plot of predictions vs real prices
- `results_summary.txt` - Detailed performance metrics
- `best_model.pth` - The trained model

## Understanding the results

The system gives you these key metrics:

- **R² Score**: How much of the price variation the model explains (higher is better)
- **MAE**: Average prediction error in currency units
- **RMSE**: Root mean square error (penalizes large errors more)
- **MAPE**: Average percentage error

## Main Features

**Simple to use**: Just run one command and get results

**Good data handling**: Properly splits data to avoid cheating, handles missing values

**Cross-validation**: Tests the model multiple times to make sure it's reliable

**Visual results**: Easy-to-understand plots show how well your model works

**Configurable**: Change settings without touching code

## Customization

Want to experiment? Try changing these in the config file:

- `hidden_layers`: Make the network bigger or smaller
- `learning_rate`: Higher = faster learning but less stable
- `epochs`: More epochs = longer training
- `total_rows`: Use more or less data

## Troubleshooting

**Out of memory?** Reduce `batch_size` or `total_rows`

**Model not learning?** Try a higher `learning_rate` or more `epochs`

**Want to see what's happening?** Check `neural_network.log` for detailed logs

## File Overview

- `main.py` - Runs everything
- `src/data/data_loader.py` - Loads and cleans the data
- `src/models/neural_network.py` - The neural network code
- `src/visualization/plotting.py` - Creates the plots
- `configs/model_config.yaml` - All the settings

That's all you need to know to get started! The system is designed to work well with sensible defaults, but you can tweak anything you want.
