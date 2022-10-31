"""
Extended Kalman filter procedure
"""

def EKF(x, P, u, z):
    x, P = predict(x, P, u)
    x, P = update(x, P, z)
    print(x[2])
    return x, P


def predict(x, P, u):
    """
    :param x: State mean (x,y,theta) [size 3 array]
    :param P: State covariance [3x3 array]
    :param u: Robot inputs (u1,u2) [size 2 array]
    :return: Predicted state mean and covariance x and P
    """
    theta = x[2]
    u1 = u[0]
    u2 = u[1]
    motion_model = np.array([DT * (RHO/2) * np.cos(theta) * (u1+u2)
                            ,DT * (RHO/2) * np.sin(theta) * (u1+u2)
                            ,DT * (RHO/L) * (u2-u1)])
    x = x + motion_model + np.random.multivariate_normal(np.zeros(3), Q)
    #print(x)
    
    Fk = np.array([[1,0,-DT * RHO * 0.5 * np.sin(theta) * (u1+u2)]
                  , [0,1,DT * RHO * 0.5 * np.cos(theta) * (u1+u2)]
                  , [0,0,1]])
    P = Fk @ P @ Fk.T + Q
    return x, P


def update(x, P, z):
    """
    :param x: State mean (x,y,theta) [size 3 array]
    :param P: State covariance [3x3 array]
    :param z: Sensor measurements [px3 array]. Each row contains range, bearing, and landmark's true (x,y) location.
    :return: Updated state mean and covariance x and P
    """
    #compute the innovation error, Hk and Rk
    true_r_phi, indexs = z[:,:2], z[:,2]
    true_r_phi = true_r_phi.reshape((-1))
    xk, yk, theta = x[0], x[1], x[2]
    #print(xk, yk, theta)
    p = len(z[:,0])
    predict_r_phi = []
    Hk = np.zeros([0,3])
    Rk = np.eye(2 * p)
    for i in range(p):
        index = int(indexs[i])
        landmark_x, landmark_y = RFID[index,:]
        #print(landmark_x, landmark_y)
        #innovation error
        predict_r = np.sqrt((landmark_x - xk)**2 + (landmark_y -yk)**2)
        predict_phi = np.arctan2(landmark_y - yk, landmark_x - xk) - theta
        #print(predict_r, predict_phi)
        predict_r_phi.append(predict_r)
        predict_r_phi.append(predict_phi)
        #Hk 
        Hk_i = np.array([[(xk-landmark_x)/predict_r, (yk-landmark_y)/predict_r, 0]
                        ,[-(yk-landmark_y)/(predict_r)**2, (xk-landmark_x)/(predict_r)**2,-1]])
        Hk = np.vstack((Hk, Hk_i))
        #Rk
        Rk[2*i:2*i+2, 2*i:2*i+2] = R
    #innovation_error
    innovation_error = true_r_phi - predict_r_phi
    for i in range(p):
        if innovation_error[2*i+1] < -np.pi:
            innovation_error[2*i+1] += 2*np.pi
        elif innovation_error[2*i+1] > np.pi:
            innovation_error[2*i+1] -= 2*np.pi
            
    if not len(Hk):
        return x, P
    
    Sk = Hk @ P @ Hk.T + Rk
    Kk = P @ Hk.T @ np.linalg.inv(Sk)
    KkHk = Kk @ Hk
    P = (np.identity(KkHk.shape[0]) - KkHk) @ P
    x = x + (Kk @ innovation_error).ravel()
    return x, P
