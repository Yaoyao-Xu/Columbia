from cgitb import reset
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os
from mpc import MPC
import time
import random
import ray
random.seed(0)
np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()


@ray.remote
def get_random_data(arm_teacher):

    X = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(), 0))
    Y = np.zeros((arm_teacher.dynamics.get_state_dim(), 0))

    count = 0

    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))

    arm_teacher.reset()
    arm_teacher.set_action(action)
    dt = args.time_step

    while count < 5:

        CURSTATE = arm_teacher.get_state() 
        action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
        for i in range(len(action)):
            action[i] = random.uniform(-5, 5)
        arm_teacher.set_action(action)
        CURACTION = np.copy(action)
        XINPUT = np.vstack((CURSTATE, CURACTION))
        arm_teacher.advance()
        YOUTPUT = arm_teacher.get_state()
        X = np.append(X, np.copy(XINPUT), axis=1)
        Y = np.append(Y, np.copy(YOUTPUT), axis=1)
        
        count += dt

    return X, Y

if __name__ == '__main__':
    args = get_args()

    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)

    # ---
    # You code goes here. Replace the X, and Y by your collected data
    # Control the arm to collect a dataset for training the forward dynamics.
    num_link = args.num_links
    X_ = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(), 0))
    Y_ = np.zeros((arm_teacher.dynamics.get_state_dim(), 0))

    result_id = []
    i = 0
    
    for _ in range(800):
        result_id.append( get_random_data.remote(arm_teacher) )
    result = ray.get(result_id)
    
    for i in range(len(result)):
        X_ = np.append(X_, result[i][0], axis=1)
        Y_ = np.append(Y_, result[i][1], axis=1) 

    print(X_.shape)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X_)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y_)
