import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Angle and Deviation X data
Angle=np.array([0,  20, 40, 60, 80, 100,120,140,160,180,\
                200,220,240,260,280,300,320,340,360])

X=np.array([0.0,0.51,0.92,1.35, 1.98, 2.6,  3.1, 3.7,  4.32, 4.85 , \
     4.29,  3.69, 3.15, 2.58, 1.95, 1.34, 0.91, 0.52,0 ])

#plot data distribution
"""
plt.figure()
plt.scatter(Angle, X)
"""

#re-adjust X
X_ra = np.copy(X)
for i in range(10, 19, 1):
    X_ra[i] = 2*X_ra[9] - X_ra[i]


#plot re-adjusted data distribution
plt.figure()
plt.scatter(Angle, X_ra)


#sklearn linear regression
model = LinearRegression(fit_intercept=True)
model.fit(Angle[:,np.newaxis], X_ra)

#New angle data
Angle_n = np.array([35.2, 42.6])

#prediction
X_n = model.predict(Angle_n[:, np.newaxis])

#rounding
X_n = np.round(X_n, 2)

#display results
print('The deviations corresponding to:')
print(Angle_n)
print('are:')
print(X_n)

#plot predicted data
plt.scatter(Angle_n, X_n, cmap='red')

#plot images.
plt.show()
