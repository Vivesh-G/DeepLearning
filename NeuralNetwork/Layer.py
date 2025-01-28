import numpy as np

#Dense Layers
class Layer:
  def __init__(self, size, activation, input_size):
    self.size = size
    self.activation = activation
    self.W = np.random.randn(self.size, input_size) * 0.01 #Weights
    self.b = np.zeros((self.size, 1)) #Biases

  def forward(self, A_prev):
    Z = np.dot(self.W, A_prev) + self.b
    A, activation_cache = self.activation.forward(Z)
    cache = (A_prev, self.W, self.b, activation_cache)
    return A, cache

  def backward(self, dA, cache):
    A_prev, W, b, activation_cache = cache
    dZ = self.activation.backward(dA, activation_cache)
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db