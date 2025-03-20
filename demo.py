import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.linalg import block_diag
from scipy.integrate import solve_ivp

def get_cube_vertices():
    """ Returns the 8 vertices of a unit cube centered at the origin. """
    return np.array([
        [-0.15, -0.15, -0.15],
        [-0.15, -0.15,  0.15],
        [-0.15,  0.15, -0.15],
        [-0.15,  0.15,  0.15],
        [ 0.15, -0.15, -0.15],
        [ 0.15, -0.15,  0.15],
        [ 0.15,  0.15, -0.15],
        [ 0.15,  0.15,  0.15]
    ]).T  # Shape (3,8)

def get_cube_faces(vertices):
    """ Returns faces of the cube given its 8 vertices. """
    return [
        [vertices[:, 0], vertices[:, 1], vertices[:, 3], vertices[:, 2]],
        [vertices[:, 4], vertices[:, 5], vertices[:, 7], vertices[:, 6]],
        [vertices[:, 0], vertices[:, 1], vertices[:, 5], vertices[:, 4]],
        [vertices[:, 2], vertices[:, 3], vertices[:, 7], vertices[:, 6]],
        [vertices[:, 0], vertices[:, 2], vertices[:, 6], vertices[:, 4]],
        [vertices[:, 1], vertices[:, 3], vertices[:, 7], vertices[:, 5]]
    ]

# Function to update both animations
def update(frame, cube, sc, state, leg1, leg2, leg3, leg4):
    # Update scatter points
    x = [state[0, frame], leg1[0, frame], leg2[0, frame], leg3[0, frame], leg4[0, frame]]
    y = [state[1, frame], leg1[1, frame], leg2[1, frame], leg3[1, frame], leg4[1, frame]]
    z = [state[2, frame], leg1[2, frame], leg2[2, frame], leg3[2, frame], leg4[2, frame]]
    sc._offsets3d = (x, y, z)
    
    # Compute rotation matrix
    phi, theta, psi = state[3, frame], state[4, frame], state[5, frame]
    R = rpy_rotation_matrix(phi, theta, psi)
    
    # Transform cube vertices
    vertices = (R @ get_cube_vertices())  # Apply rotation, shape (8,3)
    vertices = vertices + state[:3, frame].reshape(3, 1) 
    
    # Update cube faces
    faces = get_cube_faces(vertices)
    cube.set_verts(faces)
    
    return cube, sc

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

def underwater_dynamics(t,state,tau, params):
    # pose (xyz-rpy) in world frame
    x,y,z = state[0:3]
    phi,theta,psi = state[3:6]

    # velocities in body frame
    u,v,w = state[6:9]
    p,q,r = state[9:12]

    # paws positions
    P = state[12:]

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

    p1 = P[0:3]
    p2 = P[3:6]
    p3 = P[6:9]
    p4 = P[9:]

    I = np.eye(3)
    S1 = skew_symmetric_matrix(p1)
    S2 = skew_symmetric_matrix(p2)
    S3 = skew_symmetric_matrix(p3)
    S4 = skew_symmetric_matrix(p4)

    B = np.block([[I,I,I,I],[S1,S2,S3,S4]])

    F = np.linalg.pinv(B)@tau
    print(f'{F.reshape((-1,1))}\n\n')

    # q_dot = linear and angular accelerations in body frame
    # x_dot = xyz velocities and rpy rates
    q_dot = np.linalg.inv(M)@(B@F-g_vector-(D+C)@state[6:12])
    x_dot = J@state[6:12]
    P_dot = -F

    #print(f'{C_a}')

    return np.concatenate((x_dot, q_dot, P_dot))

def simulate_dynamics(t_span, initial_state, tau, params):
    sol = solve_ivp(underwater_dynamics, t_span, initial_state, args=(tau, params), method='RK45',t_eval=np.linspace(*t_span,100))
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
    p1 = np.array([1,1,-0.5])
    p2 = np.array([1,-1,-0.5])
    p3 = np.array([-1,-1,-0.5])
    p4 = np.array([-1,1,-0.5])
    t_span = (0,5)
    initial_state = np.concatenate((np.zeros(12),p1,p2,p3,p4))
    tau = np.zeros(6)
    tau[3] = 1

    sol = simulate_dynamics(t_span, initial_state, tau, params)

    # Extract time and state variables
    t = sol.t
    state = sol.y

    # Extract individual state components
    x, y, z = state[0, :], state[1, :], state[2, :]
    phi, theta, psi = state[3, :], state[4, :], state[5, :]
    u, v, w = state[6, :], state[7, :], state[8, :]
    p, q, r = state[9, :], state[10, :], state[11, :]
    leg1 = np.array([state[12,:],state[13,:],state[14,:]])
    leg2 = np.array([state[15,:],state[16,:],state[17,:]])
    leg3 = np.array([state[18,:],state[19,:],state[20,:]])
    leg4 = np.array([state[21,:],state[22,:],state[23,:]])

    # #print(state[8,:])

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot position
    plt.subplot(3, 3, 1)
    plt.plot(t, x, label='x')
    plt.plot(t, y, label='y')
    plt.plot(t, z, label='z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.title('Position vs Time')

    # Plot orientation (Euler angles)
    plt.subplot(3, 3, 2)
    plt.plot(t, phi, label='phi')
    plt.plot(t, theta, label='theta')
    plt.plot(t, psi, label='psi')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation (rad)')
    plt.legend()
    plt.title('Orientation vs Time')

    # Plot linear velocity
    plt.subplot(3, 3, 3)
    plt.plot(t, u, label='u')
    plt.plot(t, v, label='v')
    plt.plot(t, w, label='w')
    plt.xlabel('Time (s)')
    plt.ylabel('Linear Velocity (m/s)')
    plt.legend()
    plt.title('Linear Velocity vs Time')

    # Plot angular velocity
    plt.subplot(3, 3, 4)
    plt.plot(t, p, label='p')
    plt.plot(t, q, label='q')
    plt.plot(t, r, label='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.title('Angular Velocity vs Time')

    # Plot Leg positions
    plt.subplot(3, 3, 5)
    plt.plot(t, leg1[0,:], label='l1x')
    plt.plot(t, leg1[1,:], label='l1y')
    plt.plot(t, leg1[2,:], label='l1z')
    plt.xlabel('Time (s)')
    plt.ylabel('Leg1 position')
    plt.legend()
    plt.title('Leg1 Position')

    plt.subplot(3, 3, 6)
    plt.plot(t, leg2[0,:], label='l2x')
    plt.plot(t, leg2[1,:], label='l2y')
    plt.plot(t, leg2[2,:], label='l2z')
    plt.xlabel('Time (s)')
    plt.ylabel('Leg2 position')
    plt.legend()
    plt.title('Leg2 Position')

    plt.subplot(3, 3, 7)
    plt.plot(t, leg3[0,:], label='l3x')
    plt.plot(t, leg3[1,:], label='l3y')
    plt.plot(t, leg3[2,:], label='l3z')
    plt.xlabel('Time (s)')
    plt.ylabel('Leg3 position')
    plt.legend()
    plt.title('Leg3 Position')

    plt.subplot(3, 3, 8)
    plt.plot(t, leg4[0,:], label='l4x')
    plt.plot(t, leg4[1,:], label='l4y')
    plt.plot(t, leg4[2,:], label='l4z')
    plt.xlabel('Time (s)')
    plt.ylabel('Leg4 position')
    plt.legend()
    plt.title('Leg4 Position')

    plt.tight_layout()
    plt.show()

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # axis limits 
    ax.set_xlim([np.min(state[12:24, :]) - 0.1, np.max(state[12:24, :]) + 0.1])
    ax.set_ylim([np.min(state[12:24, :]) - 0.1, np.max(state[12:24, :]) + 0.1])
    ax.set_zlim([np.min(state[12:24, :]) - 0.1, np.max(state[12:24, :]) + 0.1])


    # labels
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Leg End Effectors Animation')

    # initialize scatter plot 
    sc = ax.scatter(np.zeros(5), np.zeros(5), np.zeros(5), c=['k','r', 'g', 'b', 'm'], s=50)

    # Initialize cube
    vertices = get_cube_vertices()
    faces = get_cube_faces(vertices)
    cube = Poly3DCollection(faces, color='gray', edgecolor='k', alpha=0.9)
    ax.add_collection3d(cube)

    ani = FuncAnimation(fig, update, frames=len(t), fargs=(cube, sc, state, leg1, leg2, leg3, leg4), interval=50, blit=False)
    plt.show()
    