import math
import numpy as np

def rpy_rotation_matrix(phi,theta,psi):
    Rx = np.array([[1, 0, 0],
                  [0, math.cos(phi), -math.sin(phi)],
                  [0, math.sin(phi), math.cos(phi)]])
    
    Ry = np.array([[math.cos(theta), 0, math.sin(theta)],
                  [0, 1, 0],
                  [-math.sin(theta), 0, math.cos(theta)]])
    
    Rz = np.array([[math.cos(psi), -math.sin(psi),0],
                  [math.sin(psi), math.cos(psi),0],
                  [0,0,1]])
    
    return Rz @ Ry @ Rx

def omega_to_rpy_dot_matrix(phi,theta):
    c_phi = math.cos(phi)
    s_phi = math.sin(phi)
    c_theta = math.cos(theta)
    t_theta = math.tan(theta)

    return np.array([[1,s_phi*t_theta, c_phi*t_theta],
                    [0, c_phi, -s_phi],
                    [0, s_phi/c_theta, c_phi/c_theta]])

def skew_symmetric_matrix(v):
    return np.array([[0,-v[2],v[1]],
                    [v[2],0,-v[0]],
                    [-v[1],v[0],0]])