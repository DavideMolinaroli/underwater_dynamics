import numpy as np
from demo import skew_symmetric_matrix

p1 = np.array([1,1,-0.5])
p2 = np.array([1,-1,-0.5])
p3 = np.array([-1,-1,-0.5])
p4 = np.array([-1,1,-0.5])

I = np.eye(3)
S1 = skew_symmetric_matrix(p1)
S2 = skew_symmetric_matrix(p2)
S3 = skew_symmetric_matrix(p3)
S4 = skew_symmetric_matrix(p4)

B = np.block([[I,I,I,I],[S1,S2,S3,S4]])

tau_des = np.array([0,0,0,5,0,0])

F = np.linalg.pinv(B)@tau_des

print(F.reshape((-1,1)))
