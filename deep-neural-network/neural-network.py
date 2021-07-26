import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v4a import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

print(np.__version__)

# 2 layer NN
def initialize_parameters(n_x, n_h, n_y):
  np.random.seed(1)

  W1 = np.random.randn(n_h, n_x) * 0.01
  b1 = np.zeros(shape=(n_h, 1))
  W2 = np.random.randn(n_y, n_h) * 0.01
  b2 = np.zeros(shape=(n_y, 1))

  assert(W1.shape == (n_h, n_x))
  assert(b1.shape == (n_h, 1))
  assert(W2.shape == (n_y, n_h))
  assert(b2.shape == (n_y, 1))

  parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}

  return parameters

parameters = initialize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# L-layer NN
def initialize_parameters_deep(layer_dims):
  np.random.seed(3)
  parameters = {}
  L = len(layer_dims)

  for l in range(1, L):
    parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
    parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

      
  return parameters

parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# Linear foward Propagation
def linear_forward(A, W, b):

  Z = np.dot(W, A) + b

  assert(Z.shape == (W.shape[0], A.shape[1]))
  cache = (A, W, b)
  
  return Z, cache

A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)

print("\nA_prev (n_l, n_[l-1]) = \n", linear_cache[0])
print("\nW = \n", linear_cache[1])
print("\nb = \n", linear_cache[2])
print("\n")

print("Z = " + str(Z))

# Linear-Activation Foward
def linear_activation_forward(A_prev, W, b, activation):
  if activation == "sigmoid":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = sigmoid(Z)
  elif activation == "relu":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = relu(Z)
  
  assert (A.shape == (W.shape[0], A_prev.shape[1]))
  cache = (linear_cache, activation_cache)

  return A, cache

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))

print("\nA = \n", linear_activation_cache[0][0])
print("\nW = \n", linear_activation_cache[0][1])
print("\nb = \n", linear_activation_cache[0][2])
print("\nZ = \n", linear_activation_cache[1][0])
print("\n\n")

# L-Layer Model
def L_model_forward(X, parameters):
  caches = []
  A = X
  L = len(parameters) // 2

  for l in range(1, L):
    A_prev = A 
    A, cache = linear_activation_forward(A_prev,parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
    caches.append(cache)

  AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
  caches.append(cache)

  assert(AL.shape == (1,X.shape[1]))
          
  return AL, caches

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))

for l in range(len(caches)):
    print("\nLayer ", l+1, "------------------------------------------%\n")
    print("\nA^[", l  ,"] = \n", caches[l][0][0])
    print("\nW^[", l+1,"] = \n", caches[l][0][1])
    print("\nb^[", l+1,"] = \n", caches[l][0][2])
    print("\nZ^[", l+1,"] = \n", caches[l][1][0])