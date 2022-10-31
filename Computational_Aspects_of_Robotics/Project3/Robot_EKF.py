import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def KF(x, P, z):
# IMPLEMENT THIS
  xHat = F.dot(x)
  pHat =(F.dot(P)).dot(F.T) + Q

  S = (H.dot(pHat)).dot(H.T) + R
  K = (pHat.dot(H.T)).dot(inv(S))
  yhat = z - H.dot(xHat)

  x = xHat + K.dot(yhat)
  m = K.dot(H).shape[0]
  P = (np.eye(m) - K.dot(H)).dot(pHat)

  print("the Covariance Matrix is: ", P)
  print("the Kalman Gain is: ", K)
  print("the Innovation gain is: ", S)
  print("--------------------------------------")
  return x, P

F = np.array([[1, 0.5], [0, 1]])
H = np.array([[0, 1]])
#Q = np.array([[10, 1], [1, 50]])
Q=np.array([[0.01,0.001],[0.001,0.005]])
#Q = np.array([[0.1, 0.01], [0.01, 0.05]])
R = 0.1

x = np.array([[0.8, 2]])
xhat = np.array([[2, 4]])
P = np.array([[1,0], [0,2]])

for i in range(1000):
  w = np.random.multivariate_normal(np.zeros(2), Q)
  v = np.random.normal(0, R)
  x = np.vstack((x, F @ x[-1,:] + w))
  z = H @ x[-1,:] + v

  xnew, P = KF(xhat[-1,:], P, z)
  xhat = np.vstack((xhat, xnew))

plt.plot(x[:,0], x[:,1], label = "actual state")
plt.plot(xhat[:,0], xhat[:,1], label = "predicted state")
plt.legend()
plt.xlabel('position')
plt.ylabel('velocity')
plt.show()