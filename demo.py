import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.linalg import block_diag
from scipy.integrate import solve_ivp

from algebra_utils import *
from plot_utils import *

def underwater_dynamics(t,state,tau, params):
    # pose (xyz-rpy) in world frame
    x,y,z = state[0:3]
    phi,theta,psi = state[3:6]

    # velocities in body frame
    u,v,w = state[6:9]
    p,q,r = state[9:12]

    # paws positions in world frame
    P = state[12:24]

    # Inertial params + buoyancy
    g,m,W,Ix,Iy,Iz,B = params[0]
    # Added mass params
    X_udot,Y_vdot,Z_wdot,K_pdot,M_qdot,N_rdot = params[1]
    # Damping coefficients
    Xu,Yv,Zw,Kp,Mq,Nr = params[2]

    m_rb = m*np.eye(3)
    I_rb = np.array([[Ix,0,0],
                    [0,Iy,0],
                    [0,0,Iz]])
    
    # Rigid body inertia matrix
    M_rb = block_diag(m_rb, I_rb)

    # Added mass inertia matrix
    M_a_diag = -1*np.array([X_udot,Y_vdot,Z_wdot,K_pdot,M_qdot,N_rdot])
    M_a = np.diag(M_a_diag)

    # Total inertia matrix
    M = M_rb + M_a

    #print(f'M: {M}\n')

    # Rigid body Coriolis+centripetal matrix
    C_rb_linear = skew_symmetric_matrix(np.array([m*p,m*q,m*r]))
    C_rb_rotational = skew_symmetric_matrix(np.array([-Ix*p, -Iy*q, -Iz*r]))
    C_rb = block_diag(C_rb_linear,C_rb_rotational)

    #print(f'C_rb: {C_rb}\n')

    # Added mass Coriolis+centripetal matrix
    C_a_12 = skew_symmetric_matrix(np.array([X_udot*u,Y_vdot*v,Z_wdot*w]))
    C_a_22 = skew_symmetric_matrix(np.array([K_pdot*p,M_qdot*q,N_rdot*r]))
    C_a = np.block([[np.zeros((3,3)),C_a_12],[C_a_12,C_a_22]])

    #print(f'C_a: {C_rb}\n')

    # Total Coriolis+centripetal matrix
    C = C_rb + C_a

    # Linear hydrodynamic damping
    D = -1*np.diag(np.array([Xu,Yv,Zw,Kp,Mq,Nr]))

    # RPY rotation matrix
    R = rpy_rotation_matrix(phi, theta, psi)
    
    #print(f'R: {R}\n')

    # Body angular velocity to rpy_dot transformation matrix
    T = omega_to_rpy_dot_matrix(phi,theta)

    # Body velocities to xyz-rpy derivatives transformation matrix
    J = block_diag(R,T)

    # Gravity and buoyancy (assuming CoB = origin of body frame, so no torque in induced)
    g_force = R.T @ np.array([0,0,W-B])
    g_torque = np.zeros(3)
    g_vector = np.concatenate((g_force,g_torque))

    # paws in body frame to apply torque
    p1 = R.T@P[0:3]
    p2 = R.T@P[3:6]
    p3 = R.T@P[6:9]
    p4 = R.T@P[9:]

    I = np.eye(3)
    S1 = skew_symmetric_matrix(p1)
    S2 = skew_symmetric_matrix(p2)
    S3 = skew_symmetric_matrix(p3)
    S4 = skew_symmetric_matrix(p4)

    B = np.block([[I,I,I,I],[S1,S2,S3,S4]])

    F = np.linalg.pinv(B)@tau
    
    # Paw velocity
    v_drag = -2*F/7.83*np.sqrt(7.83/(2*np.linalg.norm(F))) # needed to generate the drag force F

    #v_b_l = np.concatenate((state[6:9],state[6:9],state[6:9],state[6:9],)) # due to body linear velocity
    #v_b_w = np.concatenate((-S1@state[9:12],-S2@state[9:12],-S3@state[9:12],-S4@state[9:12])) # due to body angular velocity

    # print(f'{F.reshape((-1,1))}\n\n')

    # q_dot = linear and angular accelerations in body frame
    # x_dot = xyz velocities and rpy rates
    # P_dot = stack of paws' velocity vectors in world frame
    q_dot = np.linalg.inv(M)@(B@F-g_vector-(D+C)@state[6:12])
    x_dot = J@state[6:12]
    P_dot = block_diag(R,R,R,R)@v_drag

    print(np.linalg.norm(P_dot))

    #print(f'{C_a}')

    return np.concatenate((x_dot, q_dot, P_dot))

def simulate_dynamics(t_span, initial_state, tau, params):
    sol = solve_ivp(underwater_dynamics, t_span, initial_state, args=(tau, params), method='RK45',t_eval=np.linspace(*t_span,1000))
    return sol

if __name__ == "__main__":
    # Inertial parameters
    g = 9.81
    m = 11.5
    W = m*g 
    Ix = 0.16
    Iy = 0.16
    Iz = 0.16
    B = W
    inertial_params = np.array([g,m,W,Ix,Iy,Iz,B])

    # Added mass parameters
    X_udot = -5.5
    Y_vdot = -12.7
    Z_wdot = -14.57
    K_pdot = -0.12
    M_qdot = -0.12
    N_rdot = -0.12
    added_mass_params = np.array([X_udot,Y_vdot,Z_wdot,K_pdot,M_qdot,N_rdot])

    # Damping coefficients
    Xu = -4.03
    Yv = -6.22
    Zw = -5.18
    Kp = -0.07
    Mq = -0.07
    Nr = -0.07
    damping_coeffs = np.array([Xu,Yv,Zw,Kp,Mq,Nr])

    params = np.array([inertial_params, added_mass_params, damping_coeffs], dtype=object)
    
    # Initial paws config in body frame
    # TODO: rotate by initial state
    p1 = np.array([1,1,-0.5])
    p2 = np.array([1,-1,-0.5])
    p3 = np.array([-1,-1,-0.5])
    p4 = np.array([-1,1,-0.5])
    t_span = (0,20)
    initial_state = np.concatenate((np.zeros(12),p1,p2,p3,p4))
    initial_state[3] = 0*np.pi/180
    tau = np.zeros(6)

    # Set desired forces/torque in body frame
    tau[3] = 0.1
    tau[0] = 0

    sol = simulate_dynamics(t_span, initial_state, tau, params)

    # Extract time and state variables
    t = sol.t
    state = sol.y

    # #print(state[8,:])
    animate_body(state,tau,t, ref_frame='world')
    plot_signals(state,t)
