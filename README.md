## Basic Simulation for Underwater Dynamics

### Current Model

Considering:

![Equation](https://latex.codecogs.com/png.latex?q%20%3D%20%5Cbegin%7Bbmatrix%7D%20v%5EB_x%20%5C%5C%20v%5EB_y%20%5C%5C%20v%5EB_z%20%5C%5C%20%5Comega%5EB_x%20%5C%5C%20%5Comega%5EB_y%20%5C%5C%20%5Comega%5EB_z%20%5Cend%7Bbmatrix%7D)

(linear and angular velocity vectors expressed in the body frame)

![Equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bx%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20y%20%5C%5C%20z%20%5C%5C%20%5Cvarphi%20%5C%5C%20%5Ctheta%20%5C%5C%20%5Cpsi%20%5Cend%7Bbmatrix%7D)

(pose expressed in the world frame)

![Equation](https://latex.codecogs.com/png.latex?p_%7Bee%7D%5EB%3D%5Cbegin%7Bbmatrix%7D%20p%5EB_%7Bleg1%7D%20%5C%5C%20p%5EB_%7Bleg2%7D%20%5C%5C%20p%5EB_%7Bleg3%7D%20%5C%5C%20p%5EB_%7Bleg4%7D%20%5Cend%7Bbmatrix%7D)

(position of each paw expressed in the body frame)

![Equation](https://latex.codecogs.com/png.latex?%5Calpha%20%3D%20%5Crho%5Cpi%20r%5E2c_d)

so that a generic drag force induced by a spherical end effector of radius ![Equation](https://latex.codecogs.com/png.latex?r) whose velocity vector is ![Equation](https://latex.codecogs.com/png.latex?v) is equal to

![Equation](https://latex.codecogs.com/png.latex?%5Cfrac%7B1%7D%7B2%7D%5Crho%5Cpi%20r%5E2c_d%20v%20%7C%7Cv%7C%7C).

The system of equations:

![Equation](https://latex.codecogs.com/png.latex?%5Cbegin%7Bcases%7D%20%5Cdot%7Bq%7D%20%3D%28M_%7Brb%7D%2BM_A%29%5E%7B-1%7D%20%28-%28C_%7Brb%7D%28%7Bq%7D%29%2BC_A%28%7Bq%7D%29%29q%20-%20Dq%20-g%2B%5Ctau%29%20%5C%5C%20%5Cdot%7B%5Cmathbf%7Bx%7D%7D%20%3D%20J%28q%29q%20%5C%5C%20%5Cdot%7Bp%7D_%7Bee%7D%5EB%20%3D%20-2%20%5Cfrac%7BF%7D%7B%5Calpha%7D%5Csqrt%7B%5Cfrac%7B%5Calpha%7D%7B2%7C%7CF%7C%7C%7D%7D%20%5C%5C%20%5Ctau%20%3D%20%5Cbegin%7Bbmatrix%7D%20I%20%26%20I%20%26%20I%20%26%20I%5C%5C%20%5Bp%5EB_%7Bleg1%7D%5D_%7B%5Ctimes%7D%20%26%20%5Bp%5EB_%7Bleg2%7D%5D_%7B%5Ctimes%7D%20%26%20%5Bp%5EB_%7Bleg3%7D%5D_%7B%5Ctimes%7D%20%26%20%5Bp%5EB_%7Bleg4%7D%5D_%7B%5Ctimes%7D%20%5Cend%7Bbmatrix%7D%20F%20%3D%20B%28%7Bp%7D_%7Bee%7D%5EB%29%20F%20%5Cend%7Bcases%7D)

and 

![Equation](https://latex.codecogs.com/png.latex?F%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B12%7D)

contains the drag force generated by each leg.

Given a desired force/torque ![Equation](https://latex.codecogs.com/png.latex?%5Ctau_%7Bdes%7D) in the body frame, the minimum norm ![Equation](https://latex.codecogs.com/png.latex?F) that generates it is computed in each time instant as:

![Equation](https://latex.codecogs.com/png.latex?F%20%3D%20B%5E%7B%5Cdagger%7D%28%7Bp%7D_%7Bee%7D%5EB%29%20%5Ctau_%7Bdes%7D)
