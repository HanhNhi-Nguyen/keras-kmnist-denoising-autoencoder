# Keras KMNIST Denoising Autoencoder (TensorFlow / Keras)

This repository is a cleaned, publishable version of my *Vertiefungsprojekt* notebook.

It demonstrates a **reproducible Python ML pipeline** for **image denoising** using a small **CNN autoencoder** on the public **KMNIST** dataset.

## What this project does

- Loads **KMNIST** via `tensorflow_datasets`
- Creates *(noisy input, clean target)* pairs using **deterministic stateless Gaussian noise**
- Trains a compact **convolutional autoencoder** (Conv2D + Conv2DTranspose)
- Optionally converts the trained model to **fully INT8 quantized TFLite** using a representative dataset
- Runs a **desktop sanity-check** with a TFLite interpreter and visualizes reconstructions

## Quick start

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

### Train (float Keras)

```bash
python -m src.train
```

### Convert to INT8 TFLite + print operator list

```bash
python -m src.quantize_int8
```

### Run INT8 TFLite sanity check + plots

```bash
python -m src.evaluate_tflite_int8
```

## Notes

- Default training is **5 epochs** (kept intentionally short).
- The noise generation is **deterministic**: each sample index always receives the same noise pattern.
- The code is split from the original Colab notebook into small files for readability.
