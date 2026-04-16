"""
STRIDE 4.0 — Dense MLP Autoencoder Training

Architecture:
  Encoder: Input(6) → Dense(32) → Dense(16) → Dense(8)  ← bottleneck
  Decoder: Dense(8) → Dense(16) → Dense(32) → Dense(6)  ← reconstruction

Trained on real CMU keystroke biometrics.
Anomaly threshold: 95th percentile of training reconstruction errors.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib

print("=" * 60)
print(" S.T.R.I.D.E. 4.0 — Dense Autoencoder Training")
print("=" * 60)

# 1. Load the flat (N, 6) feature matrix
filename = "stride_sequences.npy"
print(f"\nLoading {filename}...")
try:
    X = np.load(filename)
except FileNotFoundError:
    print(f"Error: {filename} not found. Run prepare_real_dataset.py first.")
    exit(1)

print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
print(f"Features: [mean_hold, std_hold, mean_flight, std_flight, mouse_dist, error_rate]")

# 2. Normalize
print("\nNormalizing with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Build Dense Autoencoder
print("\nConstructing Dense Autoencoder topology...")
num_features = X.shape[1]

inp = Input(shape=(num_features,), name="behavioral_input")

# Encoder — compress 6D vector into 8D latent representation
x = Dense(32, activation='relu', name="enc_1")(inp)
x = Dense(16, activation='relu', name="enc_2")(x)
bottleneck = Dense(8,  activation='relu', name="bottleneck")(x)

# Decoder — reconstruct original 6D vector from latent code
x = Dense(16, activation='relu', name="dec_1")(bottleneck)
x = Dense(32, activation='relu', name="dec_2")(x)
out = Dense(num_features, activation='linear', name="reconstruction")(x)

model = Model(inp, out, name="STRIDE_DenseAE")
model.compile(optimizer='adam', loss='mse')
model.summary()

# 4. Train (autoencoder target = input itself)
print("\nTraining on real CMU keystroke data...")
history = model.fit(
    X_scaled, X_scaled,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
final_loss = history.history['val_loss'][-1]
print(f"\nFinal validation loss: {final_loss:.6f}")

# 5. Calculate the 95th percentile Anomaly Threshold
print("\nCalculating Reconstruction Error Threshold...")
X_pred      = model.predict(X_scaled, verbose=0)
mse_scores  = np.mean(np.square(X_scaled - X_pred), axis=1)  # (N,) — one MSE per sample
anomaly_threshold = np.percentile(mse_scores, 95)

print(f"  Mean MSE on training data:   {np.mean(mse_scores):.6f}")
print(f"  Std  MSE on training data:   {np.std(mse_scores):.6f}")
print(f"  95th Percentile Threshold:   {anomaly_threshold:.6f}  << ANOMALY_THRESHOLD")

# 6. Save artifacts
print("\nSaving model artifacts...")
model.save("stride_cnn.keras")   # keeping filename for backward compatibility
joblib.dump({"scaler": scaler, "anomaly_threshold": anomaly_threshold}, "stride_config.pkl")

print("\n" + "=" * 60)
print(" Training complete. S.T.R.I.D.E. 4.0 ready.")
print("=" * 60)
