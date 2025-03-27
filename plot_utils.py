import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from algebra_utils import *

def get_cube_vertices():
    return np.array([
        [-0.3, -0.3, -0.3],
        [-0.3, -0.3,  0.3],
        [-0.3,  0.3, -0.3],
        [-0.3,  0.3,  0.3],
        [ 0.3, -0.3, -0.3],
        [ 0.3, -0.3,  0.3],
        [ 0.3,  0.3, -0.3],
        [ 0.3,  0.3,  0.3]
    ]).T  # Shape (3,8)

def get_cube_faces(vertices):
    return [
        [vertices[:, 0], vertices[:, 1], vertices[:, 3], vertices[:, 2]],
        [vertices[:, 4], vertices[:, 5], vertices[:, 7], vertices[:, 6]],
        [vertices[:, 0], vertices[:, 1], vertices[:, 5], vertices[:, 4]],
        [vertices[:, 2], vertices[:, 3], vertices[:, 7], vertices[:, 6]],
        [vertices[:, 0], vertices[:, 2], vertices[:, 6], vertices[:, 4]],
        [vertices[:, 1], vertices[:, 3], vertices[:, 7], vertices[:, 5]]
    ]

def update(frame, cube, sc, state, tau, quivers, lines, ax, ref_frame):
    # Compute rotation matrix
    phi, theta, psi = state[3, frame], state[4, frame], state[5, frame]
    R = rpy_rotation_matrix(phi, theta, psi)
    
    # Paws in body frame
    leg1 = R.T @np.array([state[12,frame],state[13,frame],state[14,frame]])
    leg2 = R.T @np.array([state[15,frame],state[16,frame],state[17,frame]])
    leg3 = R.T @np.array([state[18,frame],state[19,frame],state[20,frame]])
    leg4 = R.T @np.array([state[21,frame],state[22,frame],state[23,frame]])

    I = np.eye(3)
    S1 = skew_symmetric_matrix(leg1)
    S2 = skew_symmetric_matrix(leg2)
    S3 = skew_symmetric_matrix(leg3)
    S4 = skew_symmetric_matrix(leg4)

    B = np.block([[I,I,I,I],[S1,S2,S3,S4]])

    # Compute force from desired torque (everything is expressed in the body frame)
    F = np.linalg.pinv(B)@tau

    # Extract forces 
    F1, F2, F3, F4 = F[:3], F[3:6], F[6:9], F[9:12]

    # Transform to world frame if desired
    if ref_frame == 'world':
        F1, F2, F3, F4 = R @ F1, R @ F2, R @ F3, R @ F4
        leg1, leg2, leg3, leg4 = R@leg1, R@leg2, R@leg3, R@leg4 

    # Update arrows
    legs = [leg1, leg2, leg3, leg4]
    forces = [F1, F2, F3, F4]
    for i in range(4):
        # Remove the previous quiver if it exists
        try:
            quivers[i].remove()
        except Exception as e:
            print("Warning during quiver removal:", e)
        # Create a new quiver and update the list
        quivers[i] = ax.quiver(
            legs[i][0], legs[i][1], legs[i][2],
            forces[i][0], forces[i][1], forces[i][2],
            color='r', length=1, normalize=True
        )

    com = [0,0,0]
    if ref_frame == 'world':
        com = [state[0, frame],state[1, frame],state[2, frame]]

    x = [com[0], leg1[0], leg2[0], leg3[0], leg4[0]]
    y = [com[1], leg1[1], leg2[1], leg3[1], leg4[1]]
    z = [com[2], leg1[2], leg2[2], leg3[2], leg4[2]]
    sc._offsets3d = (x, y, z)

    vertices_body = get_cube_vertices()
    if ref_frame == 'world': 
        # Transform cube vertices
        vertices = (R @ vertices_body)  # Apply rotation
        vertices = vertices + state[:3, frame].reshape(3, 1)
    else:
        vertices = vertices_body

    # Update cube faces
    faces = get_cube_faces(vertices)
    cube.set_verts(faces)

    # 0,2,4,6 are the bottom vertices
    # 6 -> leg1, 4-> leg2, 2-> leg4, 0->leg3
    remap_indices = [6,4,0,2]
    vertices_body[2,:] = vertices_body[2,:]+0.3
    midpoints = R@vertices_body
    for i, leg in enumerate(legs):
        try:
            lines[i].remove()
        except Exception as e:
            print("Warning during line removal:", e)
        
        vertex_index = remap_indices[i]
        vertex = midpoints[:,vertex_index]
        lines[i], = ax.plot([leg[0], vertex[0]],[leg[1], vertex[1]],[leg[2], vertex[2]],color='k',linestyle='-') 
    
    return cube, sc, quivers, lines

def animate_body(state,tau,t, ref_frame='world'):
    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # axis limits 
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-2, 2])

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
    cube = Poly3DCollection(faces, color='gray', edgecolor='k', alpha=0.3)
    ax.add_collection3d(cube)

    quivers = [
        ax.quiver(0, 0, 0, 0, 0, 0, color='r'),
        ax.quiver(0, 0, 0, 0, 0, 0, color='r'),
        ax.quiver(0, 0, 0, 0, 0, 0, color='r'),
        ax.quiver(0, 0, 0, 0, 0, 0, color='r')
    ]

    l0, = ax.plot([0, 0],[0,0],[0,0])
    l1, = ax.plot([0, 0],[0,0],[0,0])
    l2, = ax.plot([0, 0],[0,0],[0,0])
    l3, = ax.plot([0, 0],[0,0],[0,0])

    lines = [l0,l1,l2,l3]

    ani = FuncAnimation(fig, update, frames=len(t), fargs=(cube, sc, state, tau, quivers, lines, ax, ref_frame), interval=50, blit=False, repeat=False)
    # ani.save('animation.gif', writer='imagemagick', fps=30)
    plt.show()

def plot_signals(state,t):
    # Extract individual state components
    x, y, z = state[0, :], state[1, :], state[2, :]
    phi, theta, psi = state[3, :], state[4, :], state[5, :]
    u, v, w = state[6, :], state[7, :], state[8, :]
    p, q, r = state[9, :], state[10, :], state[11, :]
    leg1 = np.array([state[12,:],state[13,:],state[14,:]])
    leg2 = np.array([state[15,:],state[16,:],state[17,:]])
    leg3 = np.array([state[18,:],state[19,:],state[20,:]])
    leg4 = np.array([state[21,:],state[22,:],state[23,:]])

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