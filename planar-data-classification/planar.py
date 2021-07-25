import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()

plt.scatter(X[0, :], X[1, :])
plt.xlabel('x')
plt.ylabel('y')

##### DATASET #####
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

##### LOGISTIC REGRESSION #####
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.ravel())

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

##### NEURAL NETWORK STRUCTURE #####
def layer_sizes(X, Y):
  n_x = X.shape[0]
  n_h = 4
  n_y = Y.shape[0]

  return (n_x, n_h, n_y)

X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


##### INIT PARAMS #####

def initialize_parameters(n_x, n_h, n_y):
  np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.

# Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
# Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.
  W1 = np.random.randn(n_h,n_x)*0.01
  b1 = np.zeros((n_h,1))
  W2 = np.random.randn(n_y,n_h)*0.01
  b2 = np.zeros((n_y,1))

  assert (W1.shape == (n_h, n_x))
  assert (b1.shape == (n_h, 1))
  assert (W2.shape == (n_y, n_h))
  assert (b2.shape == (n_y, 1))
  
  parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}

  return parameters

n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

##### LOOP ######
def forward_propagation(X, parameters):
  # Retrieve each parameter from the dictionary "parameters"
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]

  # Implement Forward Propagation to calculate A2 (probabilities)
  Z1 = np.dot(W1,X)+b1
  A1 = np.tanh(Z1)
  Z2 = np.dot(W2,A1)+b2
  A2 = sigmoid(Z2)

  assert(A2.shape == (1, X.shape[1]))

  cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}

  return A2, cache

X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

def compute_cost(A2, Y, parameters):
  m = Y.shape[1]

  # Compute the cross-entropy cost
  logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
  cost =- np.sum(logprobs) / m

  cost = float(np.squeeze(cost)) 
  assert(isinstance(cost, float))

  return cost

A2, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

def backward_propagation(parameters, cache, X, Y):
  m = X.shape[1]

  # First, retrieve W1 and W2 from the dictionary "parameters".
  W1 = parameters["W1"]
  W2 = parameters["W2"]

  # Retrieve also A1 and A2 from dictionary "cache".
  A1 = cache["A1"]
  A2 = cache["A2"]

  # Backward propagation: calculate dW1, db1, dW2, db2. 
  dZ2 = A2-Y
  dW2 = np.dot(dZ2,A1.T) / m
  db2 = np.sum(dZ2,axis=1,keepdims=True) / m

  dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1, 2))
  dW1 = np.dot(dZ1,X.T) / m
  db1 = np.sum(dZ1,axis=1,keepdims=True) / m

  grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}

  return grads

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))