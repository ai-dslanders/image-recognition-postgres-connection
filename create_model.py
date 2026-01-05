#!/usr/bin/env python3
# ============================================================================
# CREATE PRE-TRAINED MNIST MODEL
# ============================================================================
# Run this script to create the pretrained_model.h5 file
# Usage: python create_model.py
# ============================================================================

import os
import sys

print("=" * 70)
print("MNIST PRE-TRAINED MODEL CREATOR")
print("=" * 70)

# Check TensorFlow installation
print("\n[1/5] Checking TensorFlow installation...")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"✅ TensorFlow {tf.__version__} found")
except ImportError:
    print("❌ TensorFlow not installed!")
    print("\nPlease install TensorFlow first:")
    print("  pip install tensorflow==2.15.0")
    sys.exit(1)

import numpy as np

# Create models directory
print("[2/5] Creating models directory...")
os.makedirs('./models', exist_ok=True)
print("✅ Directory ./models/ ready")

# Load MNIST dataset
print("\n[3/5] Loading MNIST dataset (60,000 training + 10,000 test images)...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"✅ Dataset loaded: train shape {x_train.shape}, test shape {x_test.shape}")

# Normalize
print("\n[4/5] Normalizing pixel values (0-255 → 0-1)...")
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print("✅ Data normalized")

# Build model
print("\n[5/5] Building and training neural network...")
print("""
Architecture:
  Input: 28x28 pixels (784 neurons when flattened)
    ↓
  Dense(128) + ReLU + Dropout(0.2)
    ↓
  Dense(64) + ReLU + Dropout(0.2)
    ↓
  Dense(10) + Softmax
    ↓
  Output: 10 classes (digits 0-9)
""")

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
print("Training model (5 epochs, ~2-3 minutes)...\n")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=1
)

# Evaluate
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n📊 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Test Loss: {test_loss:.4f}")

# Save
model_path = './models/pretrained_model.h5'
print(f"\n💾 Saving model to: {model_path}")
model.save(model_path)

file_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"✅ Model saved successfully!")
print(f"   File: {model_path}")
print(f"   Size: {file_size:.2f} MB")

print("\n" + "=" * 70)
print("✨ YOU'RE ALL SET!")
print("=" * 70)
print("""
Your pre-trained model is ready to use!

Next steps:
1. Verify file exists: ls -lh ./models/pretrained_model.h5
2. Update main.py DATABASE_URL and GOOGLE_CLIENT_ID
3. Run: python main.py
4. Visit: http://localhost:8080

The model will recognize handwritten digits 0-9 with high accuracy!
""")
print("=" * 70)
