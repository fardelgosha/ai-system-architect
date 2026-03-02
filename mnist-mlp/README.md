# MNIST Multi-Layer Perceptron

This project implements and trains a simple multi-layer perceptron (MLP) classifier on the MNIST handwritten digit dataset (28x28 grayscale images).

## Architecture
The network consists of:
- Input layer: 784 features (flattened 28x28 image)
- Hidden layer: 128 units with ReLU activation
- Output layer: 10 logits (one per digit class)

## Training Configuration
- Loss function: CrossEntropyLoss
- Optimizer: Adam (learning rate = 1e-3)
- Training duration: 1 epoch
- Mini-batch training

## Result
The trained model achieves approximately **94.7% test accuracy** on the MNIST test set.