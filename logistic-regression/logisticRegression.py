import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + 
       classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# Standardize Dataset
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# Helper functions
def sigmoid(z):
  s = 1 / (1 + np.exp(-z))

  return s

print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

# Initial Params
def initialize_with_zeros(dim):
  w = np.zeros([dim,1])
  b = 0
  assert(w.shape == (dim, 1))
  assert(isinstance(b, float) or isinstance(b, int))

  return w, b

dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))

# Forward and Backward propagation
def propagate(w, b, X, Y):
  m = X.shape[1]
  A = sigmoid(np.dot(w.T,X) + b)
  dw = 1 / m *(np.dot(X,(A - Y).T))
  db = 1 / m *(np.sum(A - Y))
  cost = -1/m * (np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1 - A).T))

  assert(dw.shape == w.shape)
  assert(db.dtype == float)
  cost = np.squeeze(cost)
  assert(cost.shape == ())

  grads = {"dw": dw,
            "db": db}

  return grads, cost

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

# Optimization
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
  costs = []
  for i in range(num_iterations):
    grads, cost = propagate(w,b,X,Y)

    dw = grads["dw"]
    db = grads["db"]

    w -= learning_rate*dw
    b -= learning_rate*db

    if i % 100 == 0:
      costs.append(cost)
    if print_cost and i % 100 == 0:
      print ("Cost after iteration %i: %f" %(i, cost))

  params = {"w": w,
            "b": b}
  grads = {"dw": dw,
            "db": db}

  return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

# predict
def predict(w, b, X):
  m = X.shape[1]
  Y_prediction = np.zeros((1,m))
  w = w.reshape(X.shape[0], 1)
  A = sigmoid(np.dot(w.T,X) + b)

  for i in range(A.shape[1]):
    if A[0][i] > 0.5:
      Y_prediction[0][i] = 1
    else:
      Y_prediction[0][i] = 0

  Y_prediction[A > 0.5] = 1.
  assert(Y_prediction.shape == (1, m))

  return Y_prediction

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))
