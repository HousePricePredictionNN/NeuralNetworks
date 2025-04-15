import matplotlib.pyplot as plt

def plot_predictions_vs_actual(y_real, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_real, y_pred, color='blue', label="Predicted vs Real", alpha=0.6)
    plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color='red', linestyle='--', label="Ideal: y = x")
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Real Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss_curve(loss_table):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_table, label="Training Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Model Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()