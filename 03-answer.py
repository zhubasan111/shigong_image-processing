import numpy as np


#original space
A =np.array([[0.126901, -0.054710, 0.938],
                  [0.076113, -0.057638, 0.942],
                  [0.074728, -0.081546, 0.895],
                  [0.125624, -0.081282, 0.893],
                  [0.156072, -0.285685, 0.827],
                  [0.019842, -0.280429, 0.851],
                  [0.092248, -0.321462, 0.763]])
 
#tranformation space
B = np.array([[0.46022323, 0.50710499, 0.28645349],
                  [0.42473236, 0.47370705, 0.28595987],
                  [0.38551146, 0.51143277, 0.28599533],
                  [0.42059597, 0.54657292, 0.28665495],
                  [0.34020177, 0.67224169, 0.13511288],
                  [0.25803548, 0.56310284, 0.13381004],
                  [0.24375798, 0.68313318, 0.13381931]])

#use Kabsch algorithm to calculate the optimal rotation matrix R and traslation vector T.
#The Kabsch algorithm minimizes the RMSD (root mean squared deviation).
#ref: https://en.wikipedia.org/wiki/Kabsch_algorithm.

#matrix processing
At, Bt = np.transpose(A), np.transpose(B)

centA = np.mean(A, axis=0)
centB = np.mean(B, axis=0)

centA = centA.reshape(-1, 1)
centB = centB.reshape(-1, 1)

Am = At - centA
Bm = Bt - centB

#Helper matrix
H = np.dot(Am, Bm.T)

#Single Value Decomposition
U, S, V = np.linalg.svd(H)

#Rotation matrix R
R = np.dot(V.T, U.T)

#Translation vector T
T = -np.dot(R, centA) + centB

#Transformation results using Kabsch algorithm.
B_kab = (np.dot(R, At) + T).T

#error between Kabsch Transformation and real data.
B_diff = B_kab - B
print()
print('the error matrix is shown below:')
print(B_diff)
print('The results show that Kabsch Transformation has high accuracy.')
print()
#new point in original space
new = np.array([-0.043618, -0.312796, 0.788])

#Kabsch Transformation of new point
new_kab = (np.dot(R, new.T) + T.T)
print('The corresponding coordinates of space 1 point:')
print(new)
print()
print('in space 2 is:')
print(new_kab)
print()