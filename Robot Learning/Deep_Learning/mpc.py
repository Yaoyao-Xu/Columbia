from operator import ne
import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_dynamics_student import ArmDynamicsStudent
from robot import Robot
from arm_gui import ArmGUI, Renderer
import argparse
import time
import math
import torch
import random
random.seed(0)
np.set_printoptions(suppress=True)

class MPC:

    def __init__(self,):
        self.control_horizon = 10
        # Define other parameters here
        self.H = 7
        self.N = 40
        self.delta = 0.2


    def get_trajectory(self, state, action, dynamics):
        trajectory = [state]

        for i in range(self.N):
            newstate = dynamics.advance(trajectory[-1], action)
            trajectory.append(np.copy(newstate))
        
        return np.array(trajectory)


    def compute_cost(self, dynamics, trajectory, goal, index):
        
        cost = 0
        for sk in trajectory[1:]:
            newpos = dynamics.compute_fk(sk)
            cost += np.linalg.norm(goal - newpos)
        return cost

    def compute_action(self, dynamics, state, goal, action):
        # Put your code here. You must return an array of shape (num_links, 1)
        #print(goal)
        n = len(action)
        trajectory = self.get_trajectory(state, action, dynamics)
 
        BestCost = self.compute_cost(dynamics, trajectory, goal, n)
        BestAction = np.copy(action)

        improving = True
        while improving:
            improving = False
            for i in range(len(action)):
                candidate = [np.copy(BestAction), np.copy(BestAction), np.copy(BestAction)]
                candidate[1][i,0] += self.delta
                candidate[2][i,0] -= self.delta

                for j in range(len(candidate)):
                    newtrajectory = self.get_trajectory(state, candidate[j], dynamics)
                    newcost = self.compute_cost(dynamics, newtrajectory, goal, n)
                    if newcost < BestCost:
                        BestCost = newcost
                        improving = True 
                        BestAction = np.copy(candidate[j])
                        
        #print(BestAction)
        return BestAction




def main(args):

    # Arm
    arm = Robot(
        ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
    )

    # Dynamics model used for control
    if args.model_path is not None:
        dynamics = ArmDynamicsStudent(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
        dynamics.init_model(args.model_path, args.num_links, args.time_step, device=torch.device("cpu"))
    else:
        # Perfectly accurate model dynamics
        dynamics = ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )

    # Controller
    controller = MPC()

    # Control loop
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    goal = np.zeros((2, 1))
    goal[0, 0] = args.xgoal
    goal[1, 0] = args.ygoal
    arm.goal = goal

    if args.gui:
        renderer = Renderer()
        time.sleep(0.25)

    dt = args.time_step
    k = 0
    while True:
        t = time.time()
        arm.advance()
        if args.gui:
            renderer.plot([(arm, "tab:blue")])
        k += 1
        time.sleep(max(0, dt - (time.time() - t)))
        if k == controller.control_horizon:
            state = arm.get_state()
            action = controller.compute_action(dynamics, state, goal, action)
            arm.set_action(action)
            k = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_links", type=int, default=3)
    parser.add_argument("--link_mass", type=float, default=0.1)
    parser.add_argument("--link_length", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--time_limit", type=float, default=5)
    parser.add_argument("--gui", action="store_const", const=True, default=False)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--xgoal", type=float, default=-1.4)
    parser.add_argument("--ygoal", type=float, default=-1.4)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
