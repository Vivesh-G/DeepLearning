import numpy as np
from Activation import Relu, sigmoid
from Layer import Layer

#Neural Network
class NeuralNetwork:
  def __init__(self, layer_dims, learning_rate):
    self.Layers = []
    self.learning_rate = learning_rate

    for i in range(1, len(layer_dims)):
      activation = Relu() if i < len(layer_dims) - 1 else sigmoid()
      self.Layers.append(Layer(layer_dims[i], activation, layer_dims[i - 1]))

  def forward_propagation(self, X):
    A = X
    caches = []
    for layer in self.Layers:
      A, cache = layer.forward(A)
      caches.append(cache)
    return A, caches

  def compute_cost(slef, AL, Y):
    m = Y.shape[1]
    e = 1e-15
    AL = np.clip(AL, e, 1 - e)
    cost = (-1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return float(np.squeeze(cost))

  def backward_propagation(self, AL, Y, caches):
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    gradients = []
    dA = dAL

    for layer, cache in reversed(list(zip(self.Layers, caches))):
      dA, dW, db = layer.backward(dA, cache)
      gradients.append((dW, db))

    return list(reversed(gradients))

  def update_parameters(self, gradients):
    for i, layer in enumerate(self.Layers):
      dW, db = gradients[i]
      layer.W -= self.learning_rate * dW
      layer.b -= self.learning_rate * db

  def train(self, X, Y, num_iterations):
    costs = []
    for i in range(num_iterations):
      AL, caches = self.forward_propagation(X)
      cost = self.compute_cost(AL, Y)
      gradients = self.backward_propagation(AL, Y, caches)
      self.update_parameters(gradients)
      costs.append(cost)
      if i % 100 == 0:
        print(f"Cost after iteration {i}: {cost}")
    return costs

  def predict(self, X, threshold = 0.5):
    AL, _ = self.forward_propagation(X)
    predictions = (AL > threshold).astype(int)
    return predictions