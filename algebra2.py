import numpy as np

# Multiplication
A = np.array([[3, 6, 7], [5, -3, 0]])
B = np.array([[1, 1], [2, 1], [3, -3]])
C = A.dot(B)
#* is used for array multiplication (multiplication of corresponding elements of two arrays) not matrix multiplication.
print("Dot multiplication 1: ", C)
 
A = np.array([[3, 6, 7], [5, -3, 0]])
B = np.array([[1, 1, 7], [2, 1, 0]])
C = np.multiply(A,B)
print("Matrix  multiplication: ", C)

# Scalar multiplcation
A = np.array([[2, 4], [5, -6]])
B = A * 7
print("Scalar multiplication", B)

# Tranpose
A = np.array([[1, 1], [2, 1], [3, -3]])
print("Matrix: ", A)
print("Tranpose: ", A.transpose())

# TENSORS
T = np.array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]],
  ])
print(T.shape)
print(T)

# Addition
A = np.array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]],
  ])
B = np.array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]],
  ])
C = A + B
print("Tensor Addition", C)