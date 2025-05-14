import pandas as pd
import os
from utils import visualize_missing_data, prepare_data_for_nn, build_neural_network
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import numpy as np 

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'combined_data.csv')
df = pd.read_csv(data_path, sep=';', encoding='utf-8', low_memory=False)
df.info()

df = df[df['distance_to_center'].notna()]
df.drop(columns=['price_per_sqm'], inplace=True)

visualize_missing_data(df)

print(df.columns.tolist())

X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_nn(df)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

input_dim = X_train.shape[1]
model = build_neural_network(input_dim)
model.summary()

# Add after your model.summary() line
print("Training model...")

# Create callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=1,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_housing_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=64,
    validation_split=0.2,  # Use 20% of training data for validation
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Evaluate on test data
print("\nEvaluating model performance...")
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss}")

# Make predictions
y_pred = model.predict(X_test)

# Calculate various performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot training history
plt.figure(figsize=(15, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot predictions vs actual values
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Housing Prices')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300)
plt.show()

# Plot error distribution
plt.figure(figsize=(10, 6))
errors = y_test - y_pred.flatten()
plt.hist(errors, bins=50)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.grid(True, alpha=0.3)
plt.savefig('error_distribution.png', dpi=300)
plt.show()

# Save timestamp for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model
model.save(f'housing_model_{timestamp}.h5')

# Save preprocessor
joblib.dump(preprocessor, f'preprocessor_{timestamp}.pkl')

print(f"Model and preprocessor saved with timestamp: {timestamp}")