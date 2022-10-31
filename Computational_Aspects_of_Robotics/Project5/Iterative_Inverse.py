from numpy import *;
import numpy as np; 
import math;
T_d=array([[0.078,-0.494,0.866,1],[0.135,-0.855,-0.500,2],[0.988,0.156,0,2],[0,0,0,1]])
Od=T_d[0:3,3]
Rx_d=T_d[0:3,0]
Ry_d=T_d[0:3,1]
Rz_d=T_d[0:3,2]
def error(Rx,Ry,Rz,On):
    delta_O= (Od-On.ravel()).reshape(-1,1)
    delta_theta= 0.5*(np.cross(Rx.ravel(),Rx_d.ravel()) + np.cross(Ry.ravel(),Ry_d.ravel()) + np.cross(Rz.ravel(),Rz_d.ravel())).reshape(-1,1)
    error=np.vstack((delta_O,delta_theta))
    return error
def Jacobian(q1,q2,q3, Lambda = 1):
  J11=sin(q1)*sin(q2)*sin(q3) - cos(q2)*sin(q1) - sin(q1) - cos(q2)*cos(q3)*sin(q1)
  J12=-cos(q1)*sin(q2)-cos(q1)*cos(q2)*sin(q3)-cos(q1)*cos(q3)*sin(q2)
  J13=-cos(q1)*cos(q2)*sin(q3)-cos(q1)*cos(q3)*sin(q2)
  J21=cos(q1) + cos(q1)*cos(q2) + cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3)
  J22=- sin(q1)*sin(q2) - cos(q2)*sin(q1)*sin(q3) - cos(q3)*sin(q1)*sin(q2)
  J23=- cos(q2)*sin(q1)*sin(q3) - cos(q3)*sin(q1)*sin(q2)
  J31=0
  J32=cos(q2) + cos(q2)*cos(q3) - sin(q2)*sin(q3)
  J33=cos(q2)*cos(q3) - sin(q2)*sin(q3)
  J=[[J11, J12, J13],[J21, J22, J23],[J31, J32, J33],[0,0,0],[0,-1,-1],[1,0,0]]
  J = np.array(J)
 
  return J
  ###Gradient descent
DELTA_ERROR = 1000
PREV_ERROR = 1000
q1, q2, q3 = 0,0,0
iteration = 0
while DELTA_ERROR > 0.00001:
  iteration += 1
  Rx, Ry, Rz, On = Rotation(q1, q2, q3)
  NEW_ERROR=error(Rx,Ry,Rz,On)
  qk= np.array([q1, q2, q3]) + 0.01*( (Jacobian(q1, q2, q3)).T.dot(NEW_ERROR)).ravel()
  q1=qk[0]
  q2=qk[1]
  q3=qk[2]
  DELTA_ERROR = abs(np.linalg.norm(NEW_ERROR) - PREV_ERROR)
  print(q1,q2,q3)
  PREV_ERROR = np.linalg.norm(NEW_ERROR)
  #break
  print("At the {} iteration, the error is {}".format(iteration, np.linalg.norm(NEW_ERROR)))
  print('------------------------------------')
  ## Newton Method
Lambda= 1
DELTA_ERROR = 1000
PREV_ERROR = 1000
q1, q2, q3 = 0,0,0
iteration = 0
while DELTA_ERROR>0.0001:
  iteration += 1
  Rx, Ry, Rz, On = Rotation(q1, q2, q3)
  NEW_ERROR=error(Rx,Ry,Rz,On)
  J = Jacobian(q1, q2, q3)
  dls_pinv = np.linalg.inv(J.T @ J + Lambda**2 * np.eye(3)) @ J.T

  qk= np.array([q1, q2, q3]) +(dls_pinv.dot(NEW_ERROR)).ravel()
  # print(J_inverse(q1, q2, q3).dot(ERROR))
  # print(qk.shape)
  q1=qk[0]
  q2=qk[1]
  q3=qk[2]
  DELTA_ERROR = abs(np.linalg.norm(NEW_ERROR) - PREV_ERROR)
  print(q1,q2,q3)
  PREV_ERROR = np.linalg.norm(NEW_ERROR)
  #break
  print("At the {} iteration, the error is {}".format(iteration, np.linalg.norm(NEW_ERROR)))
  print('------------------------------------')