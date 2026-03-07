# Metrics Plots and Effects of Variables

## General Observations

- Layer 1 contains $128 \times 784 = 100{,}352$ weight parameters, while layer 2 contains $10 \times 128 = 1{,}280$ parameters. Because the L2 norm scales with the number of parameters, the norm of the layer-1 weight matrix is significantly larger than that of layer 2.

- The norms of gradients are comparatively smaller and fluctuate during training. However, because parameter updates accumulate over many iterations, even modest gradient magnitudes can lead to substantial changes in parameter norms over time.

- Bias vectors change much less than weight matrices. This is expected because bias vectors contain far fewer parameters and their influence on activations is additive rather than multiplicative.

- The ratio $\frac{\|\nabla p\|}{\|p\|}$ for any parameter $p$ tends to decrease during training. This occurs because gradient magnitudes generally decrease as the model approaches a minimum, while parameter norms typically grow until the network reaches a stable representation scale.

- Gradients of the output layer are typically larger than those of earlier layers. This is a consequence of backpropagation, where gradients originate at the loss and propagate backward through the network. As they travel through layers, the signal often weakens.

- If the number of inputs to a layer is $n$, PyTorch typically initializes weights from the distribution $U\left(-\sqrt{\frac{1}{n}}, \sqrt{\frac{1}{n}}\right)$. This produces very small initial weights. During training, weights often increase in magnitude because the network must build stronger feature representations. However, this behavior is not universal. If weights are initialized with large values, or if weight decay is applied, the norms may instead decrease during training.

---

## Initial Variables

- Hidden layer size: **128**
- Train batch size: **64**
- Learning rate: **0.001**
- Epoch: **1**
- Training dataset size: **60000**
- Test dataset size: **10000**
- Accuracy: **94.55%**
- Training + evaluation time: **10.418 sec**
- Plots: Parameters_01, Gradient_01, Ratios_01

---

### Interpretation

- A small spike in gradient norms occurs around sample 450. This is likely caused by stochastic variation in the mini-batch gradient. Because stochastic gradient descent estimates the gradient using a subset of training samples, such fluctuations are expected.

---

### Training Dynamics

The plots Parameters_01, Gradient_01, and Ratios_01 reveal several important behaviors of neural network training:

1. **Parameter Norm Growth**

   Parameter norms gradually increase during training. This occurs because the network adjusts weights to build useful feature representations. Starting from small initial weights, training pushes parameters toward a scale where activations produce meaningful class separation.

2. **Gradient Norm Reduction**

   Gradient norms generally decrease as training progresses. This indicates that the optimizer is approaching a stationary point of the loss function where

   $$
   \nabla L(\theta) \approx 0
   $$

   In stochastic optimization, gradients rarely reach zero exactly, but instead fluctuate around a small value due to mini-batch noise.

3. **Decreasing Relative Update Size**

   The ratio

   $$
   \frac{\|\nabla p\|}{\|p\|}
   $$

   decreases during training. This indicates that parameter updates become smaller relative to parameter magnitude, suggesting that the network is converging.

4. **Layer-wise Training Behavior**

   The output layer typically exhibits larger gradient norms because it is closest to the loss function. Earlier layers receive gradients through the chain rule during backpropagation, which can reduce gradient magnitude as signals propagate backward through the network.

These observations are consistent with expected optimization dynamics in neural networks trained using mini-batch stochastic gradient descent.