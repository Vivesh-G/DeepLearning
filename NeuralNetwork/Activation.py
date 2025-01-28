import numpy as np

#Activation Functions
class sigmoid:
  def forward(self, Z):
    return 1 / (1 + np.exp(-Z)), Z

  def backward(self, dA, Z):
    forward_output, _ = self.forward(Z)
    return dA * forward_output * (1 - forward_output)

class Relu:
  def forward(self, Z):
    return np.maximum(0, Z), Z

  def backward(self, dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ