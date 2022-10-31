import math
from re import T
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import collections
from scipy.sparse.construct import random
from scipy.stats import multivariate_normal
import random

"""
Particle filtering procedures
"""
def gaussianest(x):
    size = 2
    det = np.linalg.det(R)
    norm_const = 1.0/ ( (2*math.pi) * math.pow(det,1.0/2) )
    inv = np.linalg.inv(R)
    result = np.exp(-0.5 * (x.dot(inv).dot(x.T)))
    return norm_const * result

def motionModel(x, u):
    theta = x[2]
    u1, u2 = u[0], u[1]

    deltax = DT * (RHO/2) * np.cos(theta)*(u1 + u2)
    deltay = DT * (RHO/2) * np.sin(theta)*(u1 + u2)
    deltahTheta = (RHO/L)*(u2-u1)*DT

    return np.array([deltax, deltay, deltahTheta])

def predict(x, u):
    """
    :param x: Particle state (x,y,theta) [size 3 array]
    :param u: Robot inputs (u1,u2) [size 2 array]
    :return: Particle's updated state sampled from the motion model
    """
    x = x + motionModel(x, u) + np.random.multivariate_normal(np.zeros(3), Q)

    return x


def update(x, z):
    """
    :param x: Particle state (x,y,theta) [size 3 array]
    :param z: Sensor measurements [px3 array]. Each row contains range, bearing, and landmark's true (x,y) location.
    :return: Particle's updated weight
    """
    trueR, trueB, IDList = z[:,0], z[:,1], z[:,2]
    X, Y, Theta = x[0], x[1], x[2]
    observedList = []
    for ID in IDList:
        observedList.append(int(ID))

    weight = 1
    for k in range(len(IDList)):
        index = int(IDList[k])
        trueX, trueY = RFID[index]
        r = np.sqrt( (trueX - X)**2 + (trueY - Y)**2 )
        phi = np.arctan2(trueY-Y, trueX-X) -Theta

        if r <= MAX_RANGE and index in observedList:
            deltaPhi = trueB[k] - phi
            nu = np.array([trueR[k] - r, deltaPhi])
            weight *= gaussian.pdf(nu)
        else:
            weight = 0
            break
    
    return weight


def resample(px, pw):
    """
    :param px: All current particles [3xNP array]
    :param pw: All particle weight [size NP array]
    :return: A new set of particles obtained by sampling the original particles with given weights
    """
    pw = pw[0]
    idx2data = collections.defaultdict(list)
    NORM = sum(pw)
    indexlist = []
    for i in range(len(pw)):
        pw[i] /= NORM
        idx2data[i] = [px[0][i], px[1][i], px[2][i]]
        indexlist.append(i)
    x = np.random.choice(indexlist, len(pw), p=pw)
    for i in range(NP):
        px[0][i] = idx2data[x[i]][0]
        px[1][i] = idx2data[x[i]][1]
        px[2][i] = idx2data[x[i]][2]
    
    return px