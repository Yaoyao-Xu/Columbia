import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy import optimize 


def total_potential(q, q_goal, zita, ita, q_star, obst_q_1,obst_q_2, obst_r):

    point_1 = np.array(obst_q_1) + (np.array(q) - np.array(obst_q_1)) / np.linalg.norm(np.array(q) - np.array(obst_q_1)) * obst_r
    obst_dist_1 = dist(q, point_1)
    point_2 = np.array(obst_q_2) + (np.array(q) - np.array(obst_q_2)) / np.linalg.norm(np.array(q) - np.array(obst_q_2)) * obst_r
    obst_dist_2 = dist(q, point_2)

    u_a = 0.5 * zita * dist(q, q_goal)**2
    u_r_1 = 0.5 * ita * (1/obst_dist_1 - 1/q_star)**2
    u_r_2 = 0.5 * ita * (1/obst_dist_2 - 1/q_star)**2

    return u_a + u_r_1+u_r_2


def grad(q, q_goal, d_star, zita, ita, q_star, obst_q_1,obst_q_2, obst_r):

    point_1 = np.array(obst_q_1) + (np.array(q) - np.array(obst_q_1)) / np.linalg.norm(np.array(q) - np.array(obst_q_1)) * obst_r
    point_2 = np.array(obst_q_2) + (np.array(q) - np.array(obst_q_2)) / np.linalg.norm(np.array(q) - np.array(obst_q_2)) * obst_r
    obst_dist_1 = dist(q, point_1)
    obst_dist_2 = dist(q, point_2)


    # attractive grad piece-wise function
    if dist(q, q_goal) <= d_star:
        grad_a = zita * np.array([q[0] - q_goal[0], q[1] - q_goal[1]])
    else:
        grad_a = d_star / dist(q, q_goal) * zita * np.array([q[0] - q_goal[0], q[1] - q_goal[1]])    

    # repulsive grad piece-wise function
    if obst_dist_1 > q_star:
        grad_r_1 =  np.array([0, 0]) 
    else:
        grad_obst = (np.array(q) - np.array(point_1)) / obst_dist_1
        grad_r_1 = ita/(obst_dist_1**2) * (1/q_star - 1/obst_dist_1) * grad_obst 
    
    # repulsive grad piece-wise function    
    if obst_dist_2 > q_star:
        grad_r_2 =  np.array([0, 0]) 
    else:
        grad_obst = (np.array(q) - np.array(point_2)) / obst_dist_2
        grad_r_2 = ita/(obst_dist_2**2) * (1/q_star - 1/obst_dist_2) * grad_obst 
    
    
    
    return grad_a + grad_r_1+grad_r_2


def dist(q0, q1):
  
    return np.sqrt((q1[0] - q0[0])**2 + (q1[1] - q0[1])**2)

potential = total_potential(q=(0.5, -0.5), q_goal=(0, 0), zita=1, ita=1, q_star=1, obst_q_1=[2, 0], obst_q_2=[-2, 0],obst_r=1)
gradient = grad(q=[0.5, -0.5], q_goal=[0, 0], d_star=1, zita=1, ita=1, q_star=1, obst_q_1=[2, 0], obst_q_2=[-2, 0],obst_r=1)
print("Solution to Problem 4 Part (b): Potential={}".format(potential))
print("Solution to Problem 4 Part (b): Gradient={}".format(gradient))


