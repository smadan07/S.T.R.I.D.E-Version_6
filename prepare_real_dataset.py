"""
STRIDE 4.0 — Dataset Preparation
Source: CMU Keystroke Dynamics Benchmark (DSL-StrongPasswordData.csv)
        51 subjects × 400 attempts = 20,400 real human samples.

Feature Engineering:
  REAL (from CMU):  mean_hold, std_hold, mean_flight, std_flight
  SYNTHESIZED:      mouse_dist, error_rate  (no equivalent in keyboard-only dataset)

Output: stride_sequences.npy with shape (N, 6) — flat feature matrix.
"""
import urllib.request
import pandas as pd
import numpy as np

print("Downloading CMU Keystroke Dynamics Benchmark Dataset...")
url = "https://raw.githubusercontent.com/njanakiev/keystroke-biometrics/master/data/DSL-StrongPasswordData.csv"
try:
    urllib.request.urlretrieve(url, "DSL-StrongPasswordData.csv")
    print("Download complete.")
except Exception as e:
    print(f"Download failed (using cached file if available): {e}")

print("Processing dataset...")
df = pd.read_csv("DSL-StrongPasswordData.csv")
np.random.seed(42)

# --- Extract ALL hold and flight columns (not just 2) ---
# The password is .tie5Roanl — 11 keys, 10 transitions
hold_cols   = [c for c in df.columns if c.startswith("H.")]   # 11 hold time columns
flight_cols = [c for c in df.columns if c.startswith("UD.")]  # 10 up-down (flight) columns

print(f"Using {len(hold_cols)} hold columns + {len(flight_cols)} flight columns from CMU.")

# Convert to milliseconds
hold_ms   = df[hold_cols].values   * 1000.0
flight_ms = df[flight_cols].values * 1000.0

# Clip outliers (negatives in flight = key overlap, which is valid behavior)
hold_ms   = np.clip(hold_ms,   10.0,  500.0)
flight_ms = np.clip(flight_ms, -50.0, 800.0)

# --- Feature Engineering: statistics across all key pairs per row ---
# Each row represents one complete password attempt by one subject.
# Computing mean + std across the 11 hold values and 10 flight values
# captures BOTH the average speed AND the consistency of that subject's rhythm.
mean_hold   = np.mean(hold_ms, axis=1)   # ms — average key hold duration
std_hold    = np.std(hold_ms, axis=1)    # ms — tightness of hold rhythm (low = consistent typist)
mean_flight = np.mean(flight_ms, axis=1) # ms — average inter-key gap
std_flight  = np.std(flight_ms, axis=1)  # ms — flight time consistency

# --- Synthesize mouse features (correlated to typing speed) ---
# Fast typists (low flight time) tend to have more aggressive mouse movements.
speed_factor = np.clip(mean_flight / 120.0, 0.3, 3.0)
mouse_dist   = 380.0 * speed_factor + np.random.normal(0, 60.0, len(df))
mouse_dist   = np.clip(mouse_dist, 50.0, 2000.0)

# Error rate: low baseline, slightly higher for faster typists
error_rate = np.random.exponential(0.3, len(df)) * speed_factor
error_rate = np.clip(error_rate, 0.0, 5.0)

# --- Stack into final (N, 6) feature matrix ---
# Feature order must match what main.py sends during live inference:
# [mean_hold, std_hold, mean_flight, std_flight, mouse_dist, error_rate]
data = np.column_stack([mean_hold, std_hold, mean_flight, std_flight, mouse_dist, error_rate])

print(f"\nGenerated feature matrix: {data.shape}")
print(f"\n{'Feature':<22} {'Min':>8} {'Mean':>8} {'Max':>8}")
print("-" * 50)
names = ["mean_hold (ms)", "std_hold (ms)", "mean_flight (ms)", "std_flight (ms)", "mouse_dist (px)", "error_rate"]
for i, name in enumerate(names):
    print(f"{name:<22} {data[:,i].min():>8.2f} {data[:,i].mean():>8.2f} {data[:,i].max():>8.2f}")

np.save("stride_sequences.npy", data)
print(f"\nSaved stride_sequences.npy — shape: {data.shape}")
print("Keyboard features (hold + flight): 100% real CMU measurements.")
print("Mouse features: synthesized, correlated to keystroke speed.")
