# Numpy Tutorial, with sources for each question for future consults on the problems. TODO -> Organize this file in a .org (Org mode) file

import numpy as np
from numpy import linalg as LA
import pandas as pd
import random
from numpy.polynomial.polynomial import polygrid2d
from numpy.polynomial.polynomial import polygrid3d
from ast import literal_eval
from PIL import Image
import matplotlib.pyplot as plt

##### NumPy Array


# How to create an empty and a full NumPy array? https://www.geeksforgeeks.org/how-to-create-an-empty-and-a-full-numpy-array/
empa = np.empty((3, 4), dtype=int)
print("Empty Array")
print(empa)

flla = np.full([3, 3], 55, dtype=int)
print("\n Full Array")
print(flla)

# Create a Numpy array filled with all zeros https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
np.zeros((5,), dtype=int)

# Create a Numpy array filled with all ones https://www.geeksforgeeks.org/create-a-numpy-array-filled-with-all-ones/
b = np.ones([3, 3], dtype = int) 

# Check whether a Numpy array contains a specified row https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-155.php
num = np.arange(20)
arr1 = np.reshape(num, [4, 5])
print("Original array:")
print(arr1)
print([0, 1, 2, 3, 4] in arr1.tolist())
print([0, 1, 2, 3, 5] in arr1.tolist())
print([15, 16, 17, 18, 19] in arr1.tolist())

# How to Remove rows in Numpy array that contains non-numeric values? https://www.geeksforgeeks.org/how-to-remove-rows-in-numpy-array-that-contains-non-numeric-values/
n_arr = np.array([[10.5, 22.5, 3.8],
                  [41, np.nan, np.nan]])
  
print("Given array:")
print(n_arr)
  
print("\nRemove all rows containing non-numeric elements")
print(n_arr[~np.isnan(n_arr).any(axis=1)])

# Remove single-dimensional entries from the shape of an array https://www.w3resource.com/numpy/manipulation/squeeze.php#:~:text=The%20squeeze()%20function%20is,the%20shape%20of%20an%20array.&text=Input%20data.&text=Selects%20a%20subset%20of%20the,one%2C%20an%20error%20is%20raised.
a = np.array([[[0], [2], [4]]])
np.squeeze(a, axis=0).shape

# Find the number of occurrences of a sequence in a NumPy array https://www.geeksforgeeks.org/find-the-number-of-occurrences-of-a-sequence-in-a-numpy-array/
arr = np.array([[2, 8, 9, 4], 
                   [9, 4, 9, 4],
                   [4, 5, 9, 7],
                   [2, 9, 4, 3]])
  
output = repr(arr).count("9, 4")

# Find the most frequent value in a NumPy array https://www.geeksforgeeks.org/find-the-most-frequent-value-in-a-numpy-array/
x = np.array([1,2,3,4,5,1,2,1,1,1])
print(np.bincount(x).argmax())

# Combining a one and a two-dimensional NumPy Array https://www.geeksforgeeks.org/combining-a-one-and-a-two-dimensional-numpy-array/
num_1d = np.arange(5)   
num_2d = np.arange(10).reshape(2,5) 
for a, b in np.nditer([num_1d, num_2d]):
    print("%d:%d" % (a, b),)

# How to build an array of all combinations of two NumPy arrays? https://www.geeksforgeeks.org/how-to-build-an-array-of-all-combinations-of-two-numpy-arrays/
np.array(np.meshgrid([1, 2, 3], [4, 5], [6, 7])).T.reshape(-1,3)

# How to add a border around a NumPy array? https://www.geeksforgeeks.org/how-to-add-a-border-around-a-numpy-array/
array = np.ones((2, 2))
array = np.pad(array, pad_width=1, mode='constant',
               constant_values=0)

# How to compare two NumPy arrays?
np.array_equal([1, 2], [1, 2])

# How to check whether specified values are present in NumPy array? https://www.geeksforgeeks.org/how-to-check-whether-specified-values-are-present-in-numpy-array/
n_array = np.array([[2, 3, 0],
                    [4, 1, 6]])

# How to get all 2D diagonals of a 3D NumPy array? https://www.geeksforgeeks.org/how-to-get-all-2d-diagonals-of-a-3d-numpy-array/
np_array = np.arange(3*4*5).reshape(3,4,5)
result = np.diagonal(np_array, axis1=1, axis2=2)

# Flatten a Matrix in Python using NumPy https://www.geeksforgeeks.org/flatten-a-matrix-in-python-using-numpy/#:~:text=flatten()%20function%20we%20can,to%20one%20dimension%20in%20python.&text=order%3A'C'%20means%20to,%2C%20row%2Dmajor%20order%20otherwise.
gfg = np.array([[2, 3], [4, 5]])
flat_gfg = gfg.flatten()

# Flatten a 2d numpy array into 1d array https://www.geeksforgeeks.org/python-flatten-a-2d-numpy-array-into-1d-array/
ini_array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
print("initial array", str(ini_array1))
result = ini_array1.flatten()

# Move axes of an array to new positions https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-52.php
x = np.zeros((2, 3, 4))
print(np.moveaxis(x, 0, -1).shape)
print(np.moveaxis(x, -1, 0).shape)

# Interchange two axes of an array https://www.geeksforgeeks.org/numpy-swapaxes-function-python/
arr = np.array([[2, 4, 6]])
gfg = np.swapaxes(arr, 0, 1)
print (gfg)

# NumPy – Fibonacci Series using Binet Formula https://www.geeksforgeeks.org/numpy-fibonacci-series-using-binet-formula/
a = np.arange(1, 11)
lengthA = len(a)
sqrtFive = np.sqrt(5)
alpha = (1 + sqrtFive) / 2
beta = (1 - sqrtFive) / 2
Fn = np.rint(((alpha ** a) - (beta ** a)) / (sqrtFive))
print("The first {} numbers of Fibonacci series are {} . ".format(lengthA, Fn))

# Counts the number of non-zero values in the array https://www.geeksforgeeks.org/numpy-count_nonzero-method-python/
arr = [[0, 1, 2, 3, 0], [0, 5, 6, 0, 7]]
gfg = np.count_nonzero(arr)
print (gfg) 

# Count the number of elements along a given axis https://www.geeksforgeeks.org/numpy-size-function-python/
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(np.size(arr, 0))
print(np.size(arr, 1))

# Trim the leading and/or trailing zeros from a 1-D array https://www.geeksforgeeks.org/numpy-trim_zeros-in-python/
gfg = np.array((0, 0, 0, 0, 1, 5, 7, 0, 6, 2, 9, 0, 10, 0, 0))
res = np.trim_zeros(gfg)
print(res)

# Change data type of given numpy array https://www.tutorialspoint.com/change-data-type-of-given-numpy-array-in-python#:~:text=We%20have%20a%20method%20called,()%20method%20of%20numpy%20array.
array = np.array([1.5, 2.6, 3.7, 4.8, 5.9])
array = array.astype(np.int32)

# Reverse a numpy array https://www.geeksforgeeks.org/python-reverse-a-numpy-array/
ini_array = np.array([1, 2, 3, 6, 4, 5])

print("initial array", str(ini_array))
print("type of ini_array", type(ini_array))

res = np.flipud(ini_array)

print("final array", str(res))

# How to make a NumPy array read-only? https://www.geeksforgeeks.org/how-to-make-a-numpy-array-read-only/
a = np.zeros(11)
print("Before any change ")
print(a)
  
a[1] = 2
print("Before after first change ")
print(a)
  
a.flags.writeable = False
print("After making array immutable on attempting  second change ")
a[1] = 7


###### Questions on NumPy Matrix


# Get the maximum value from given matrix https://numpy.org/doc/stable/reference/generated/numpy.matrix.max.html
x = np.matrix(np.arange(12).reshape((3,4)));x
([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
x.max()

# Get the minimum value from given matrix https://numpy.org/doc/stable/reference/generated/numpy.matrix.min.html

x = -np.matrix(np.arange(12).reshape((3,4))); x
([[  0,  -1,  -2,  -3],
        [ -4,  -5,  -6,  -7],
        [ -8,  -9, -10, -11]])
x.min()


# Find the number of rows and columns of a given matrix using NumPy https://www.w3resource.com/python-exercises/numpy/basic/numpy-basic-exercise-26.php
m= np.arange(10,22).reshape((3, 4))
print("Original matrix:")
print(m)
print("Number of rows and columns of the said matrix:")
print(m.shape)

# Select the elements from a given matrix https://numpy.org/doc/stable/reference/generated/numpy.select.html
x = np.arange(10)
condlist = [x<3, x>5]
choicelist = [x, x**2]
np.select(condlist, choicelist)

# Find the sum of values in a matrix https://numpy.org/doc/stable/reference/generated/numpy.matrix.sum.html
x = np.matrix([[1, 2], [4, 3]])
x.sum()

# Calculate the sum of the diagonal elements of a NumPy array https://www.geeksforgeeks.org/calculate-the-sum-of-the-diagonal-elements-of-a-numpy-array/
n_array = np.array([[55, 25, 15],
                    [30, 44, 2],
                    [11, 45, 77]])
print("Numpy Matrix is:")
print(n_array)
trace = np.trace(n_array)
print("\nTrace of given 3X3 matrix:")
print(trace)

# Adding and Subtracting Matrices in Python https://www.geeksforgeeks.org/adding-and-subtracting-matrices-in-python/
A = np.array([[1, 2], [3, 4]])
B = np.array([[4, 5], [6, 7]])
  
print("Printing elements of first matrix")
print(A)
print("Printing elements of second matrix")
print(B)
print("Addition of two matrix")
print(np.add(A, B))

# Ways to add row/columns in numpy array https://www.geeksforgeeks.org/python-ways-to-add-row-columns-in-numpy-array/
ini_array = np.array([[1, 2, 3], [45, 4, 7], [9, 6, 10]])
print("initial_array : ", str(ini_array))

column_to_be_added = np.array([1, 2, 3])
result = np.hstack((ini_array, np.atleast_2d(column_to_be_added).T))
 
print ("resultant array", str(result))

# Matrix Multiplication in NumPy https://numpy.org/doc/stable/reference/generated/numpy.dot.html
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
np.dot(a, b)

# Get the eigen values of a matrix https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html
x = np.random.random()
Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])

# How to Calculate the determinant of a matrix using NumPy? https://www.geeksforgeeks.org/how-to-calculate-the-determinant-of-a-matrix-using-numpy/
n_array = np.array([[50, 29], [30, 44]])
  
print("Numpy Matrix is:")
print(n_array)
det = np.linalg.det(n_array)
  
print("\nDeterminant of given 2X2 matrix:")
print(int(det))

# How to inverse a matrix using NumPy https://www.geeksforgeeks.org/how-to-inverse-a-matrix-using-numpy/
A = np.array([[6, 1, 1],
              [4, -2, 5],
              [2, 8, 7]])
  
print(np.linalg.inv(A))

# How to count the frequency of unique values in NumPy array? https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-94.php
a = np.array( [10,10,20,10,20,20,20,30, 30,50,40,40] )
print("Original array:")
print(a)
unique_elements, counts_elements = np.unique(a, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Multiply matrices of complex numbers using NumPy in Python https://www.geeksforgeeks.org/multiply-matrices-of-complex-numbers-using-numpy-in-python/
x = np.array([2+3j, 4+5j])
print("Printing First matrix:")
print(x)
  
y = np.array([8+7j, 5+6j])
print("Printing Second matrix:")
print(y)
  
z = np.vdot(x, y)
print("Product of first and second matrices are:")
print(z)

# Compute the outer product of two given vectors using NumPy in Python https://www.geeksforgeeks.org/compute-the-outer-product-of-two-given-vectors-using-numpy-in-python/
array1 = np.array([6,2])
array2 = np.array([2,5])
print("Original 1-D arrays:")
print(array1)
print(array2)
  
print("Outer Product of the two array is:")
result = np.outer(array1, array2)
print(result)

# Calculate inner, outer, and cross products of matrices and vectors using NumPy https://www.geeksforgeeks.org/calculate-inner-outer-and-cross-products-of-matrices-and-vectors-using-numpy/
a = np.array([2, 6])
b = np.array([3, 10])
print("Vectors :")
print("a = ", a)
print("\nb = ", b)
  
print("\nInner product of vectors a and b =")
print(np.inner(a, b))
  
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])
print("\nMatrices :")
print("x =", x)
print("\ny =", y)
print("\nInner product of matrices x and y =")
print(np.inner(x, y))

# Compute the covariance matrix of two given NumPy arrays https://www.geeksforgeeks.org/compute-the-covariance-matrix-of-two-given-numpy-arrays/
array1 = np.array([0, 1, 1])
array2 = np.array([2, 2, 1])
  
print("\nCovariance matrix of the said arrays:\n",
      np.cov(array1, array2))

# Convert covariance matrix to correlation matrix using Python https://www.geeksforgeeks.org/convert-covariance-matrix-to-correlation-matrix-using-python/
dataset = pd.read_csv("iris.csv")
dataset.head()

# Compute the Kronecker product of two mulitdimension NumPy arrays https://www.geeksforgeeks.org/compute-the-kronecker-product-of-two-mulitdimension-numpy-arrays/
array1 = np.array([[1, 2], [3, 4]])  
array2 = np.array([[5, 6], [7, 8]])
  
kroneckerProduct = np.kron(array1, array2)
print(kroneckerProduct)

# Convert the matrix into a list https://numpy.org/doc/stable/reference/generated/numpy.matrix.tolist.html
x = np.matrix(np.arange(12).reshape((3,4))); x
([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
x.tolist()


###### Questions on NumPy Indexing


# Replace NumPy array elements that doesn’t satisfy the given condition https://www.geeksforgeeks.org/replace-numpy-array-elements-that-doesnt-satisfy-the-given-condition/
n_arr = np.array([75.42436315, 42.48558583, 60.32924763])
print("Given array:")
print(n_arr)
  
print("\nReplace all elements of array which are greater than 50. to 15.50")
n_arr[n_arr > 50.] = 15.50
  
print("New array :\n")
print(n_arr)

# Return the indices of elements where the given condition is satisfied https://www.geeksforgeeks.org/numpy-where-in-python/
a = np.array([[1, 2, 3], [4, 5, 6]])
  
print(a)
print ('Indices of elements <4')
  
b = np.where(a<4)
print(b)
  
print("Elements which are <4")
print(a[b])

# Replace NaN values with average of columns https://www.geeksforgeeks.org/python-replace-nan-values-with-average-of-columns/
ini_array = np.array([[1.3, 2.5, 3.6, np.nan], 
                      [2.6, 3.3, np.nan, 5.5],
                      [2.1, 3.2, 5.4, 6.5]])
  
print ("initial array", ini_array)
col_mean = np.nanmean(ini_array, axis = 0)
  
print ("columns mean", str(col_mean))
inds = np.where(np.isnan(ini_array))
  
ini_array[inds] = np.take(col_mean, inds[1])
print ("final array", ini_array)

# Replace negative value with zero in numpy array https://www.geeksforgeeks.org/python-replace-negative-value-with-zero-in-numpy-array/
ini_array1 = np.array([1, 2, -3, 4, -5, -6])
  
result = np.where(ini_array1<0, 0, ini_array1)
print("New resulting array: ", result)

# How to get values of an NumPy array at certain index positions? https://www.geeksforgeeks.org/how-to-get-values-of-an-numpy-array-at-certain-index-positions/
a1 = np.array([11, 10, 22, 30, 33])
print("Array 1 :")
print(a1)
  
a2 = np.array([1, 15, 60])
print("Array 2 :")
print(a2)
  
print("\nTake 1 and 15 from Array 2 and put them in\
1st and 5th position of Array 1")
  
a1.put([0, 4], a2)
  
print("Resultant Array :")
print(a1)

# Find indices of elements equal to zero in a NumPy array https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-115.php
nums = np.array([1,0,2,0,3,0,4,5,6,7,8])
print("Original array:")
print(nums)
print("Indices of elements equal to zero of the said array:")
result = np.where(nums == 0)[0]
print(result)

# How to Remove columns in Numpy array that contains non-numeric values? https://www.geeksforgeeks.org/how-to-remove-columns-in-numpy-array-that-contains-non-numeric-values/
n_arr = np.array([[10.5, 22.5, np.nan],
                  [41, 52.5, np.nan]])
  
print("Given array:")
print(n_arr)
  
print("\nRemove all columns containing non-numeric elements ")
print(n_arr[:, ~np.isnan(n_arr).any(axis=0)])

# How to access different rows of a multidimensional NumPy array?
arr = np.array([[10, 20, 30], 
                [40, 5, 66], 
                [70, 88, 94]])
  
print("Given Array :")
print(arr)
  
# Access the First and Last rows of array https://www.geeksforgeeks.org/how-to-access-different-rows-of-a-multidimensional-numpy-array/
res_arr = arr[[0,2]]
print("\nAccessed Rows :")
print(res_arr)

# Get row numbers of NumPy array having element larger than X https://www.geeksforgeeks.org/get-row-numbers-of-numpy-array-having-element-larger-than-x/
arr = np.array([[1, 2, 3, 4, 5],
                  [10, -3, 30, 4, 5],
                  [3, 2, 5, -4, 5],
                  [9, 7, 3, 6, 5] 
                 ])
X = 6
print("Given Array:\n", arr)
output  = np.where(np.any(arr > X,
                                axis = 1))
print("Result:\n", output)

# Get filled the diagonals of NumPy array https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html
a = np.zeros((3, 3), int)
np.fill_diagonal(a, 5)

# Check elements present in the NumPy array https://www.kite.com/python/answers/how-to-check-if-a-value-exists-in-numpy-array#:~:text=Use%20Python%20keyword%20in%20to,contains%20num%20and%20False%20otherwise.
num = 40
arr = np.array([[1, 30],
                [4, 40]])

if num in arr:
    print(True)
else:
    print(False)

# Combined array index by index (not sure about this one :think:) https://stackoverflow.com/questions/21233224/how-to-logically-combine-integer-indices-in-numpy
a = np.random.rand(10, 20, 30)

idx1 = np.where(a>0.2)
idx2 = np.where(a<0.4)

ridx1 = np.ravel_multi_index(idx1, a.shape)
ridx2 = np.ravel_multi_index(idx2, a.shape)
ridx = np.intersect1d(ridx1, ridx2)
idx = np.unravel_index(ridx, a.shape)

np.allclose(a[idx], a[(a>0.2) & (a<0.4)])


##### Questions on NumPy Linear Algebra


# Find a matrix or vector norm using NumPy https://www.geeksforgeeks.org/find-a-matrix-or-vector-norm-using-numpy/
vec = np.arange(10)
vec_norm = np.linalg.norm(vec)
 
print("Vector norm:")
print(vec_norm)

# Calculate the QR decomposition of a given matrix using NumPy https://www.geeksforgeeks.org/calculate-the-qr-decomposition-of-a-given-matrix-using-numpy/
matrix1 = np.array([[1, 2, 3], [3, 4, 5]])
q, r = np.linalg.qr(matrix1)
print('\nQ:\n', q)
print('\nR:\n', r)

# Compute the condition number of a given matrix using NumPy https://www.geeksforgeeks.org/compute-the-condition-number-of-a-given-matrix-using-numpy/
matrix = np.array([[4, 2], [3, 1]])

print("Original matrix:")
print(matrix)
  
result =  np.linalg.cond(matrix)
  
print("Condition number of the matrix:")
print(result)

# Compute the eigenvalues and right eigenvectors of a given square array using NumPy? https://www.geeksforgeeks.org/how-to-compute-the-eigenvalues-and-right-eigenvectors-of-a-given-square-array-using-numpy/
m = np.array([[1, 2, 3],
              [2, 3, 4],
              [4, 5, 6]])
  
print("Printing the Original square array:\n",
      m)
  
w, v = np.linalg.eig(m)
  
print("Printing the Eigen values of the given square array:\n",
      w)
  
print("Printing Right eigenvectors of the given square array:\n",
      v)

# Calculate the Euclidean distance using NumPy https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
point1 = np.array((1, 2, 3))
point2 = np.array((1, 1, 1))
 
dist = np.linalg.norm(point1 - point2)
 
print(dist)


###### Questions on NumPy Random


# Create a Numpy array with random values https://numpy.org/doc/1.20/reference/random/generated/numpy.random.rand.html
np.random.rand(3,2)

# How to choose elements from the list with different probability using NumPy? https://www.geeksforgeeks.org/how-to-choose-elements-from-the-list-with-different-probability-using-numpy/
num_list = [10, 20, 30, 40, 50]
number = np.random.choice(num_list)
print(number)

# How to get weighted random choice in Python? https://www.geeksforgeeks.org/how-to-get-weighted-random-choice-in-python/
sampleList = [100, 200, 300, 400, 500]
  
randomList = random.choices(
  sampleList, weights=(10, 20, 30, 40, 50), k=5)
  
print(randomList)

# Generate Random Numbers From The Uniform Distribution using NumPy https://www.geeksforgeeks.org/generate-random-numbers-from-the-uniform-distribution-using-numpy/
r = np.random.uniform(size=4)
print(r)

# Get Random Elements form geometric distribution https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html
z = np.random.geometric(p=0.35, size=10000)

# Get Random elements from Laplace distribution https://numpy.org/doc/1.20/reference/random/generated/numpy.random.laplace.html
loc, scale = 0., 1.
s = np.random.laplace(loc, scale, 1000)

# Return a Matrix of random values from a uniform distribution
s = np.random.uniform(-1,0,1000)

# Return a Matrix of random values from a Gaussian distribution https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 1000)


###### Questions on NumPy Sorting and Searching


# How to get the indices of the sorted array using NumPy in Python? https://www.w3resource.com/python-exercises/numpy/python-numpy-sorting-and-searching-exercise-5.php
student_id = np.array([1023, 5202, 6230, 1671, 1682, 5241, 4532])
print("Original array:")
print(student_id)
i = np.argsort(student_id)
print("Indices of the sorted elements of a given array:")
print(i)

# Finding the k smallest values of a NumPy array https://www.geeksforgeeks.org/finding-the-k-smallest-values-of-a-numpy-array/
arr = np.array([23, 12, 1, 3, 4, 5, 6])
print("The Original Array Content")
print(arr)
  
k = 4
  
arr1 = np.sort(arr)
  
print(k, "smallest elements of the array")
print(arr1[:k])

# How to get the n-largest values of an array using NumPy? https://www.kite.com/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
numbers = np.array([1, 3, 2, 4])
n = 2
indices = (-numbers).argsort()[:n]
print(indices)

# Sort the values in a matrix https://numpy.org/doc/stable/reference/generated/numpy.matrix.sort.html
a = np.array([[1,4], [3,1]])
a.sort(axis=1)

# Filter out integers from float numpy array  https://www.geeksforgeeks.org/python-filter-out-integers-from-float-numpy-array/
ini_array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
print ("initial array : ", str(ini_array))
result = ini_array[ini_array != ini_array.astype(int)]
print ("final array", result)

# Find the indices into a sorted array  https://www.geeksforgeeks.org/numpy-searchsorted-in-python/#:~:text=searchsorted()%20function%20is%20used,find%20the%20required%20insertion%20indices.
in_arr = [2, 3, 4, 5, 6]
print ("Input array : ", in_arr)
  
num = 4
print("The number which we want to insert : ", num) 
    
out_ind = np.searchsorted(in_arr, num) 
print ("Output indices to maintain sorted array : ", out_ind)


##### Questions on NumPy Mathematics


# How to get element-wise true division of an array using Numpy? https://www.geeksforgeeks.org/how-to-get-element-wise-true-division-of-an-array-using-numpy/
x = np.arange(5)
  
print("Original array:", 
      x)
rslt = np.true_divide(x, 4)
  
print("After the element-wise division:", 
      rslt)

# How to calculate the element-wise absolute value of NumPy array? https://www.geeksforgeeks.org/how-to-calculate-the-element-wise-absolute-value-of-numpy-array/
array = np.array([1, -2, 3])
  
print("Given array:\n", array)
rslt = np.absolute(array)
  
print("Absolute array:\n", rslt)

# Compute the negative of the NumPy array https://www.geeksforgeeks.org/numpy-negative-in-python/
in_num = 10
  
print ("Input  number : ", in_num)
    
out_num = np.negative(in_num) 
print ("negative of input number : ", out_num) 

# Multiply 2d numpy array corresponding to 1d array https://www.geeksforgeeks.org/python-multiply-2d-numpy-array-corresponding-to-1d-array/
ini_array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
ini_array2 = np.array([0, 2, 3])
  
print("initial array", str(ini_array1))
result = ini_array1 * ini_array2[:, np.newaxis]
  
# printing result
print("New resulting array: ", result)

# Computes the inner product of two arrays https://numpy.org/doc/stable/reference/generated/numpy.inner.html
np.inner(a, b) = sum(a[:]*b[:])

# Compute the nth percentile of the NumPy array https://www.geeksforgeeks.org/numpy-percentile-in-python/
arr = [20, 2, 7, 1, 34]
print("arr : ", arr)
print("50th percentile of arr : ",
       np.percentile(arr, 50))
print("25th percentile of arr : ",
       np.percentile(arr, 25))
print("75th percentile of arr : ",
       np.percentile(arr, 75))

# Calculate the n-th order discrete difference along the given axis https://www.geeksforgeeks.org/numpy-diff-in-python/
arr = np.array([1, 3, 4, 7, 9])
   
print("Input array  : ", arr)
print("First order difference  : ", np.diff(arr))
print("Second order difference : ", np.diff(arr, n = 2))
print("Third order difference  : ", np.diff(arr, n = 3))

# Calculate the sum of all columns in a 2D NumPy array https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-152.php
num = np.arange(36)
arr1 = np.reshape(num, [4, 9])
print("Original array:")
print(arr1)
result  = arr1.sum(axis=0)
print("\nSum of all columns:")
print(result)

# Calculate average values of two given NumPy arrays https://www.geeksforgeeks.org/calculate-average-values-of-two-given-numpy-arrays/
arr1 = np.array([3, 4])
arr2 = np.array([1, 0])
avg = (arr1 + arr2) / 2

print("Average of NumPy arrays:\n",
      avg)

# How to compute numerical negative value for all elements in a given NumPy array? https://www.geeksforgeeks.org/how-to-compute-numerical-negative-value-for-all-elements-in-a-given-numpy-array/
x = np.array([-1, -2, -3,
              1, 2, 3, 0])

print("Printing the Original array:",
      x)
r1 = np.negative(x)

print("Printing the negative value of the given array:",
      r1)

# How to get the floor, ceiling and truncated values of the elements of a numpy array? https://www.w3resource.com/python-exercises/numpy/python-numpy-math-exercise-10.php
x = np.array([-1.6, -1.5, -0.3, 0.1, 1.4, 1.8, 2.0])
print("Original array:")
print(x)
print("Floor values of the above array elements:")
print(np.floor(x))
print("Ceil values of the above array elements:")
print(np.ceil(x))
print("Truncated values of the above array elements:")
print(np.trunc(x))

# How to round elements of the NumPy array to the nearest integer? https://numpy.org/doc/stable/reference/generated/numpy.rint.html
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
np.rint(a)

# Find the round off the values of the given matrix https://www.geeksforgeeks.org/python-numpy-matrix-round/
gfg = np.matrix('[6.4, 1.3; 12.7, 32.3]')
           
# applying matrix.round() method
geeks = gfg.round()     
print(geeks)

# Determine the positive square-root of an array https://www.geeksforgeeks.org/numpy-sqrt-in-python/
arr1 = np.sqrt([1, 4, 9, 16])
arr2 = np.sqrt([6, 10, 18])
  
print("square-root of an array1  : ", arr1)
print("square-root of an array2  : ", arr2)

# Evaluate Einstein’s summation convention of two multidimensional NumPy arrays https://www.geeksforgeeks.org/evaluate-einsteins-summation-convention-of-two-multidimensional-numpy-arrays/
matrix1 = np.array([[1, 2], [0, 2]])
matrix2 = np.array([[0, 1], [3, 4]])
  
print("Original matrix:")
print(matrix1)
print(matrix2)
  
result = np.einsum("mk,kn", matrix1, matrix2)
  
print("Einstein’s summation convention of the two matrix:")
print(result)


##### Questions on NumPy Statistics


# Compute the median of the flattened NumPy array https://www.geeksforgeeks.org/compute-the-median-of-the-flattened-numpy-array/
x_odd = np.array([1, 2, 3, 4, 5, 6, 7])
print("\nPrinting the Original array:")
print(x_odd)
  
med_odd = np.median(x_odd)
print("\nMedian of the array that contains \
odd no of elements:")
print(med_odd)

# Find Mean of a List of Numpy Array https://numpy.org/doc/stable/reference/generated/numpy.mean.html
a = np.array([[1, 2], [3, 4]])
np.mean(a)

# Calculate the mean of array ignoring the NaN value https://www.geeksforgeeks.org/python-numpy-nanmean-function/
arr = np.array([[20, 15, 37], [47, 13, np.nan]])
print("Shape of array is", arr.shape)
print("Mean of array without using nanmean function:",
                                           np.mean(arr))
   
print("Using nanmean function:", np.nanmean(arr))

# Get the mean value from given matrix https://numpy.org/doc/stable/reference/generated/numpy.matrix.mean.html
x = np.matrix(np.arange(12).reshape((3, 4)))
x.mean()

# Compute the variance of the NumPy array https://numpy.org/doc/stable/reference/generated/numpy.var.html#:~:text=The%20variance%20is%20the%20average,N%20%3D%20len(x)%20.
a = np.array([[1, 2], [3, 4]])
np.var(a)

# Compute the standard deviation of the NumPy array https://numpy.org/doc/stable/reference/generated/numpy.std.html#:~:text=The%20standard%20deviation%20is%20the,N%20%3D%20len(x)%20.
a = np.array([[1, 2], [3, 4]])
np.std(a)

# Compute pearson product-moment correlation coefficients of two given NumPy arrays https://www.geeksforgeeks.org/compute-pearson-product-moment-correlation-coefficients-of-two-given-numpy-arrays/
array1 = np.array([0, 1, 2])
array2 = np.array([3, 4, 5])
rslt = np.corrcoef(array1, array2)
  
print(rslt)

# Calculate the mean across dimension in a 2D NumPy array https://www.w3resource.com/python-exercises/numpy/python-numpy-math-exercise-19.php
x = np.array([[10, 30], [20, 60]])
print("Original array:")
print(x)
print("Mean of each column:")
print(x.mean(axis=0))
print("Mean of each row:")
print(x.mean(axis=1))

# Calculate the average, variance and standard deviation in Python using NumPy https://www.geeksforgeeks.org/calculate-the-average-variance-and-standard-deviation-in-python-using-numpy/
list = [2, 4, 4, 4, 5, 5, 7, 9]
print(np.average(list))

# Describe a NumPy Array in Python https://www.geeksforgeeks.org/describe-a-numpy-array-in-python/
arr = np.array([4, 5, 8, 5, 6, 4,
                9, 2, 4, 3, 6])    
mean = np.mean(arr)
median = np.median(arr)
  
print("Array =", arr)
print("Mean =", mean)
print("Median =", median)


##### Questions on Polynomial


# Define a polynomial function https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html
p = np.poly1d([1, 2, 3])
print(np.poly1d(p))

# How to add one polynomial to another using NumPy in Python? https://www.geeksforgeeks.org/how-to-add-one-polynomial-to-another-using-numpy-in-python/
px = (5,-2,5)
#q(x) = 2(x**2) + (-5)x +2
qx = (2,-5,2) 
#add the polynomials
rx = np.polynomial.polynomial.polyadd(px,qx)
#print the resultant polynomial
print(rx)

# How to subtract one polynomial to another using NumPy in Python? https://www.geeksforgeeks.org/how-to-subtract-one-polynomial-to-another-using-numpy-in-python/
#define the polynomials
#p(x) = 5(x**2) + (-2)x +5
px = (5,-2,5)
#q(x) = 2(x**2) + (-5)x +2
qx = (2,-5,2)
#subtract the polynomials
rx = np.polynomial.polynomial.polysub(px,qx)
#print the resultant polynomial
print(rx)

# How to multiply a polynomial to another using NumPy in Python? https://www.geeksforgeeks.org/how-to-multiply-a-polynomial-to-another-using-numpy-in-python/
#define the polynomials
#p(x) = 5(x**2) + (-2)x +5
px = (5, -2, 5)
#q(x) = 2(x**2) + (-5)x +2
qx = (2, -5, 2)
#mul the polynomials
rx = np.polynomial.polynomial.polymul(px, qx)
#print the resultant polynomial
print(rx)

# How to divide a polynomial to another using NumPy in Python? https://www.geeksforgeeks.org/how-to-divide-a-polynomial-to-another-using-numpy-in-python/
#p(x) = 5(x**2) + (-2)x +5
px = (5, -2, 5)
#q(x) = 2(x**2) + (-5)x +2
qx = (2, -5, 2)
#mul the polynomials
rx = np.polynomial.polynomial.polydiv(px, qx)
#print the resultant polynomial
print(rx)

# Find the roots of the polynomials using NumPy https://www.geeksforgeeks.org/find-the-roots-of-the-polynomials-using-numpy/
coeff = [1, 2, 1]
print(np.roots(coeff))

# Evaluate a 2-D polynomial series on the Cartesian product https://www.geeksforgeeks.org/python-numpy-np-polygrid2d-method/
c = np.array([[1, 3, 5], [2, 4, 6]]) 
ans = polygrid2d([7, 9], [8, 10], c)
print(ans)

# Evaluate a 3-D polynomial series on the Cartesian product
c = np.array([[1, 3, 5], [2, 4, 6], [10, 11, 12]]) 
ans = polygrid3d([7, 9], [8, 10], [5, 6], c)
print(ans)


##### Questions on NumPy Strings


# Repeat all the elements of a NumPy array of strings https://www.geeksforgeeks.org/repeat-all-the-elements-of-a-numpy-array-of-strings/
arr = np.array(['Akash', 'Rohit', 'Ayush', 
                'Dhruv', 'Radhika'], dtype = np.str)
print("Original Array :")
print(arr)
new_array = np.char.multiply(arr, 3)
print("\nNew array :")
print(new_array)

# How to split the element of a given NumPy array with spaces? https://www.geeksforgeeks.org/how-to-split-the-element-of-a-given-numpy-array-with-spaces/
array = np.array(['PHP C# Python C Java C++'], dtype=np.str)
print(array)
sparr = np.char.split(array)
print(sparr)

# How to insert a space between characters of all the elements of a given NumPy array? https://www.geeksforgeeks.org/how-to-insert-a-space-between-characters-of-all-the-elements-of-a-given-numpy-array/
x = np.array(["geeks", "for", "geeks"],
             dtype=np.str)
print("Printing the Original Array:")
print(x)
r = np.char.join(" ", x)
print("Printing the array after inserting space\
between the elements")
print(r)

# Find the length of each string element in the Numpy array https://www.geeksforgeeks.org/find-the-length-of-each-string-element-in-the-numpy-array/
arr = np.array(['New York', 'Lisbon', 'Beijing', 'Quebec'])
print(arr)

# Swap the case of an array of string https://www.geeksforgeeks.org/numpy-string-operations-swapcase-function/
in_arr = np.array(['P4Q R', '4q Rp', 'Q Rp4', 'rp4q'])
print ("input array : ", in_arr)
out_arr = np.char.swapcase(in_arr)
print ("output swapcasecased array :", out_arr)

# Change the case to uppercase of elements of an array https://numpy.org/doc/stable//reference/generated/numpy.char.upper.html
c = np.array(['a1b c', '1bca', 'bca1'])
np.char.upper(c)

# Change the case to lowercase of elements of an array https://numpy.org/doc/stable/reference/generated/numpy.char.lower.html
c = np.array(['A1B C', '1BCA', 'BCA1']); c
np.char.lower(c)

# Join String by a seperator https://www.geeksforgeeks.org/numpy-string-operations-join-function/
in_arr = np.array(['Python', 'Numpy', 'Pandas'])
print ("Input original array : ", in_arr) 
sep = np.array(['-', '+', '*'])
out_arr = np.core.defchararray.join(sep, in_arr)
print ("Output joined array: ", out_arr)

# Check if two same shaped string arrayss one by one https://www.geeksforgeeks.org/numpy-string-operations-equal-function/
in_arr1 = np.array('numpy')
print ("1st Input array : ", in_arr1)
in_arr2 = np.array('numpy')
print ("2nd Input array : ", in_arr2)  
out_arr = np.char.equal(in_arr1, in_arr2)
print ("Output array: ", out_arr)

# Count the number of substrings in an array https://www.geeksforgeeks.org/numpy-string-operations-count-function/
in_arr = np.array(['Sayantan', '  Sayan  ', 'Sayansubhra'])
print ("Input array : ", in_arr)  
out_arr = np.char.count(in_arr, sub ='an')
print ("Output array: ", out_arr) 

# Find the lowest index of the substring in an array https://www.w3resource.com/python-exercises/numpy/python-numpy-string-exercise-16.php
x1 = np.array(['Python', 'PHP', 'JS', 'EXAMPLES', 'HTML'], dtype=np.str)
print("\nOriginal Array:")
print(x1)
print("count the lowest index of ‘P’:")
r = np.char.find(x1, "P")
print(r)

# Get the boolean array when values end with a particular character https://www.geeksforgeeks.org/python-numpy-np-char-endswith-method/
a = np.array(['geeks', 'for', 'geeks'])
gfg = np.char.endswith(a, 'ks')  
print(gfg)

##### More Questions on NumPy

# Different ways to convert a Python dictionary to a NumPy array https://www.geeksforgeeks.org/different-ways-to-convert-a-python-dictionary-to-a-numpy-array/
name_list = """{
   "column0": {"First_Name": "Akash",
   "Second_Name": "kumar", "Interest": "Coding"},
                  
   "column1": {"First_Name": "Ayush",
   "Second_Name": "Sharma", "Interest": "Cricket"},
     
   "column2": {"First_Name": "Diksha",
   "Second_Name": "Sharma","Interest": "Reading"},
     
   "column3": {"First_Name":" Priyanka",
   "Second_Name": "Kumari", "Interest": "Dancing"}
     
  }"""
print("Type of name_list created:\n",
      type(name_list))
t = literal_eval(name_list)
print("\nPrinting the original Name_list dictionary:\n",
      t)
  
print("Type of original dictionary:\n",
      type(t))
result_nparra = np.array([[v[j] for j in ['First_Name', 'Second_Name',
                                          'Interest']] for k, v in t.items()])
  
print("\nConverted ndarray from the Original dictionary:\n",
      result_nparra)

print("Type:\n", type(result_nparra))

# How to convert a list and tuple into NumPy arrays? https://www.geeksforgeeks.org/how-to-convert-a-list-and-tuple-into-numpy-arrays/
list1 = [3, 4, 5, 6]
print(type(list1))
print(list1)
array1 = np.asarray(list1)
print(type(array1))
print(array1)
tuple1 = ([8, 4, 6], [1, 2, 3])
print(type(tuple1))
print(tuple1)
array2 = np.asarray(tuple1)
print(type(array2))
print(array2)

# Ways to convert array of strings to array of floats https://www.delftstack.com/howto/numpy/numpy-convert-string-array-to-float-array/
stringArray = np.array(["1.000", "1.235", "0.000125", "2", "55", "-12.35", "0", "-0.00025"])
floatArray = stringArray.astype(float)
print(stringArray)
print(floatArray)

# Convert a NumPy array into a csv file https://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file
pd.DataFrame(np_array).to_csv("path/to/file.csv")

# How to Convert an image to NumPy array and save it to CSV file using Python? https://www.geeksforgeeks.org/how-to-convert-an-image-to-numpy-array-and-saveit-to-csv-file-using-python/
img = Image.open('geeksforgeeks.jpg')
imageToMatrice = gfg.asarray(img)
print(imageToMatrice.shape)

# How to save a NumPy array to a text file? kite.com/python/answers/how-to-save-a-numpy-array-to-a-text-file-in-python
original_array = np.loadtxt("test.txt").reshape(4, 2)

# Load data from a text file https://www.geeksforgeeks.org/import-text-files-into-numpy-arrays/
File_data = np.loadtxt("example1.txt", dtype=int)
print(File_data)

# Plot line graph from NumPy array https://www.geeksforgeeks.org/plot-line-graph-from-numpy-array/
x = np.arange(1, 11)
y = x * x

plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color ="red")
plt.show()

# Create Histogram using NumPy https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
a = np.arange(5)
hist, bin_edges = np.histogram(a, density=True)
