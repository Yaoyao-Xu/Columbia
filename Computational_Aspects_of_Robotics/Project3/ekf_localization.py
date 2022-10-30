"""
COMS 4733 Fall 2021 Homework 4
Scaffolding code for localization using an extended Kalman filter
Inspired by a similar example on the PythonRobotics project
https://pythonrobotics.readthedocs.io/en/latest/
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# "True" robot noise (filters do NOT know these)
WHEEL1_NOISE = 0.05
WHEEL2_NOISE = 0.1
BEARING_SENSOR_NOISE = np.deg2rad(1.0)

# Physical robot parameters (filters do know these)
RHO = 1
L = 1
#MAX_RANGE = 10.0    # maximum observation range
MAX_RANGE = 18.0
# RFID positions [x, y]
RFID = np.array([[-5.0, -5.0],
                 [10.0, 0.0],
                 [10.0, 10.0],
                 [0.0, 15.0],
                 [-5.0, 20.0]])

# Covariances used by the estimators
Q = np.diag([0.1, 0.1, np.deg2rad(1.0)]) ** 2
R = np.diag([0.4, np.deg2rad(1.0)]) ** 2

# Other parameters
DT = 0.1            # time interval [s]
SIM_TIME = 30.0     # simulation time [s]

# Plot limits
XLIM = [-20,20]
YLIM = [-10,30]
show_animation = True


"""
Robot physics
"""
def input(time, x):
    # Control inputs to the robot at a given time for a given state
    psi1dot = 3.7
    psi2dot = 4.0
    return np.array([psi1dot, psi2dot])

def move(x, u):
    # Physical motion model of the robot: x_k = f(x_{k-1}, u_k)
    # Incorporates imperfections in the wheels
    theta = x[2]
    psi1dot = u[0] * (1 + np.random.rand() * WHEEL1_NOISE)
    psi2dot = u[1] * (1 + np.random.rand() * WHEEL2_NOISE)

    velocity = np.array([RHO/2 * np.cos(theta) * (psi1dot+psi2dot),
                         RHO/2 * np.sin(theta) * (psi1dot+psi2dot),
                         RHO/L * (psi2dot - psi1dot)])

    return x + DT * velocity

def measure(x):
    # Physical measurement model of the robot: z_k = h(x_k)
    # Incorporates imperfections in both range and bearing sensors
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - x[0]
        dy = RFID[i, 1] - x[1]
        r = math.sqrt(dx**2 + dy**2)
        phi = math.atan2(dy, dx) - x[2]

        if r <= MAX_RANGE:
            zi = np.array([[np.round(r),
                            phi + np.random.randn() * BEARING_SENSOR_NOISE,
                            i]])
            z = np.vstack((z, zi))

    return z


"""
Extended Kalman filter procedure
"""
def normalize(theta):
    if theta > 2*math.pi:
        theta -= 2*math.pi
    return theta

def EKF(x, P, u, z):
    x, P = predict(x, P, u)
    x, P = update(x, P, z)
    #print(x[2])
    return x, P

def motionModel(x, u):
    theta = x[2]
    u1, u2 = u[0], u[1]

    deltax = DT * (RHO/2) * np.cos(theta)*(u1 + u2)
    deltay = DT * (RHO/2) * np.sin(theta)*(u1 + u2)
    deltahTheta = (RHO/L)*(u2-u1)*DT

    return np.array([deltax, deltay, deltahTheta])

def observationModel(x, z): # return y_bar
    trueR, trueB, ID = z[:,0], z[:,1], z[:,2]
    trueSensor = []
    PredictSensor = []
    p = len(z)

    X, Y, Theta = x[0], x[1], x[2]

    for i in range(len(ID)):
        trueSensor.append(trueR[i])
        trueSensor.append(trueB[i])

        index = int(ID[i])
        trueX, trueY = RFID[index]
        r = np.sqrt( (trueX - X)**2 + (trueY - Y)**2 )
        phi = np.arctan2(trueY-Y, trueX-X) -Theta
        PredictSensor.append(r)
        PredictSensor.append(phi)
        

    trueSensor = np.array(trueSensor).reshape(-1,1)
    PredictSensor = np.array(PredictSensor).reshape(-1,1)

    nu =  trueSensor - PredictSensor
    # '''
    # Careful with the computation of the innovation. In this case it represents the difference 
    # between two bearings, which means you need to handle wraparound cases (-270° is equivalent to 90°). 
    # The innovation should never exceed pi in magnitude.
    # '''
    for i in range(p):
        while nu[2*i+1] < -np.pi:
            nu[2*i+1] += 2 * np.pi
					
        while nu[2*i+1] > np.pi:
				
            nu[2*i+1] -= 2*np.pi

    return nu

def observationJacobian(x, z):
    X, Y = x[0], x[1]
    ID = z[:,2]
    p = len(z)
    H = np.array([])
    Rk = np.eye(2*p)
    for i in range(len(ID)):
        index = int(ID[i])
        trueX, trueY = RFID[index]
        rdist = np.sqrt( (trueX - X)**2 + (trueY - Y)**2 )
        Hi = np.array( [ [(X-trueX)/rdist, (Y-trueY)/rdist, 0], [-(Y-trueY)/(rdist**2), (X-trueX)/(rdist**2), -1] ] )
        if not len(H):
            H = Hi
        else:
            H = np.vstack((H, Hi))
        Rk[2*i:2*i+2 , 2*i:2*i+2] = R
    return H, Rk

def motionJacobian(x, u):
    u1, u2 = u[0] ,u[1]
    Theta = x[2]


    dx = np.array([1, 0, (-RHO/2)*np.sin(Theta)*(u1 + u2)*DT])
    dy = np.array([0, 1, (RHO/2)*np.cos(Theta)*(u1 + u2)*DT])
    dtheta = np.array([0,0,1])
    return np.vstack((dx, dy, dtheta))




def predict(x, P, u):
    """
    :param x: State mean (x,y,theta) [size 3 array]
    :param P: State covariance [3x3 array]
    :param u: Robot inputs (u1,u2) [size 2 array]
    :return: Predicted state mean and covariance x and P
    """
    
    x = x + motionModel(x, u) + np.random.multivariate_normal(np.zeros(3), Q)
    F = motionJacobian(x, u)
    P = F.dot(P).dot(F.T) + Q


    return x, P



def update(x, P, z):
    """
    :param x: State mean (x,y,theta) [size 3 array]
    :param P: State covariance [3x3 array]
    :param z: Sensor measurements [px3 array]. Each row contains range, bearing, and landmark's true (x,y) location.
    :return: Updated state mean and covariance x and P
    """
    
    ybar = observationModel(x, z)
    H , Rk = observationJacobian(x, z)
    if not len(H):
        return x, P
    S = H.dot(P).dot(H.T)+ Rk
    K = P.dot(H.T).dot(np.linalg.inv(S))

    KH = K.dot(H)
    I = np.identity(KH.shape[0])

    x = x + (K.dot(ybar)).ravel()
    P = (I-KH).dot(P)

    return x, P


def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0]).flatten()
    py = np.array(fx[1, :] + xEst[1]).flatten()
    plt.plot(px, py, "--g")


def main():
    time = 0.0

    # Initialize state and covariance
    x_est = np.array([20,20,20])
    x_true = np.zeros(3)
    P = np.eye(3)

    # State history
    h_x_est = x_est.T
    h_x_true = x_true.T

    while time <= SIM_TIME:
        time += DT
        u = input(time, x_true)
        x_true = move(x_true, u)
        z = measure(x_true)
        x_est, P = EKF(x_est, P, u, z)

        # store data history
        h_x_est = np.vstack((h_x_est, x_est))
        h_x_true = np.vstack((h_x_true, x_true))

        if show_animation:
            plt.cla()

            for i in range(len(z[:,0])):
                plt.plot([x_true[0], RFID[int(z[i,2]),0]], [x_true[1], RFID[int(z[i,2]),1]], "-k")
            plt.plot(RFID[:,0], RFID[:,1], "*k")
            plt.plot(np.array(h_x_true[:,0]).flatten(),
                     np.array(h_x_true[:,1]).flatten(), "-b")
            plt.plot(np.array(h_x_est[:,0]).flatten(),
                     np.array(h_x_est[:,1]).flatten(), "-r")
            plot_covariance_ellipse(x_est, P)

            plt.axis("equal")
            plt.xlim(XLIM)
            plt.ylim(YLIM)
            plt.grid(True)
            plt.pause(0.001)

    plt.figure()
    errors = np.abs(h_x_true - h_x_est)
    plt.plot(errors)
    dth = errors[:,2] % (2*np.pi)
    errors[:,2] = np.amin(np.array([2*np.pi-dth, dth]), axis=0)
    plt.legend(['x error', 'y error', 'th error'])
    plt.xlabel('time')
    plt.ylabel('error magnitude')
    plt.ylim([0,1.5])
    plt.show()


if __name__ == '__main__':
    main()
