import torch
import tensorflow as tf
import jax
import ml_dtypes
print("--- System Check ---")
# 1. PyTorch
print(f"PyTorch GPU: {torch.backends.mps.is_available() or torch.cuda.is_available()}")
# 2. TensorFlow
gpu_tf = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPU: {len(gpu_tf) > 0}")
# 3. JAX (Robust Method)
try: print(f"JAX Backend: {jax.devices()[0].platform.upper()}")
except Exception as e: print(f"JAX Backend Check Failed: {e}")
# 4. Dependency Check
print(f"ml_dtypes version: {ml_dtypes.__version__}")
print("--- Setup Complete ---")