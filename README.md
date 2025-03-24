## Basic Simulation for Underwater Dynamics

### Current Model

  Considering
  
  $$q = \begin{bmatrix}
      v^B_x \\ v^B_y \\ v^B_z \\ \omega^B_x \\ \omega^B_y \\ \omega^B_z
  \end{bmatrix}$$ (linear and angular velocity vectors expressed in body frame)\
  
  $$\mathbf{x} = \begin{bmatrix}
      x \\ y \\ z \\ \varphi \\ \theta \\ \psi
  \end{bmatrix}$$  (pose expressed in world frame)\

  $$p_{ee}^B=\begin{bmatrix}
      p^B_{leg1} \\
      p^B_{leg2} \\
      p^B_{leg3} \\ 
      p^B_{leg4}
  \end{bmatrix}$$ (position of each paw expressed in body frame)

  $\alpha = \rho\pi r^2c_d$, so that a generic drag force induced by a spherical end effector of radius $r$ whose velocity vector is $v$ is equal to $\frac{1}{2}\rho\pi r^2c_d v||v||$.

  $$
  \begin{cases}
      \dot{q} =(M_{rb}+M_A)^{-1} (-(C_{rb}({q})+C_A({q}))q - Dq -g+\tau) \\
      \dot{\mathbf{x}} = J(q)q \\
      \dot{p}_{ee}^B = -2 \frac{F}{\alpha}\sqrt{\frac{\alpha}{2||F||}} \\
      \tau = \begin{bmatrix}
          I & I & I & I\\
          [p^B_{leg1}]_{\times} & [p^B_{leg2}]_{\times} & [p^B_{leg3}]_{\times} & [p^B_{leg4}]_{\times} 
      \end{bmatrix} F = B({p}_{ee}^B) F
  \end{cases}
  $$
  
  and $F \in R^{12}$ contains the drag force generated by each leg.
  Given a desired force/torque $\tau_{des}$ in body frame, the minimum norm $F$ that generates it is computed, in each time instant, as $F = B^{\dagger}({p}_{ee}^B) \tau_{des}$
