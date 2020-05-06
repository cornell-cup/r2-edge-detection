import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# First joint angle (shoulder) represented by (r (=222mm), theta_1, phi_1)
r_1 = 222
theta_1 = 60
phi_1 = 45

# Second joint angle (elbow) represented by (r (=300mm), theta_2, phi_2)
r_2 = 300
theta_2 = 45
phi_2 = 45

origin = [0, 0, 0]
shoulder_coord = [r_1 * np.sin(theta_1) * np.cos(phi_1), r_1 *
                  np.sin(theta_1) * np.sin(phi_1), r_1 * np.cos(theta_1)]
elbow_coord = [r_2 * np.sin(theta_2) * np.cos(phi_2), r_2 *
               np.sin(theta_2) * np.sin(phi_2), r_2 * np.cos(theta_2)]
point_to_reach = [5, 6, 7]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = [origin[0], shoulder_coord[0], elbow_coord[0]]
Y = [origin[1], shoulder_coord[1], elbow_coord[1]]
Z = [origin[2], shoulder_coord[2], elbow_coord[2]]


ax.plot(X, Y, Z)
ax.scatter(point_to_reach[0], point_to_reach[1], point_to_reach[2], c='green')
ax.scatter(origin[0], origin[1], origin[2], c='red')
ax.scatter(shoulder_coord[0], shoulder_coord[1], shoulder_coord[2], c='red')
ax.scatter(elbow_coord[0], elbow_coord[1], elbow_coord[2], c='red')
ax.text(origin[0], origin[1], origin[2], 'Shoulder', 'x', color='red')
ax.text(point_to_reach[0], point_to_reach[1],
        point_to_reach[2], 'Point to Reach', 'x', color='green')
ax.text(shoulder_coord[0], shoulder_coord[1],
        shoulder_coord[2], 'Elbow', 'x', color='red')
ax.text(elbow_coord[0], elbow_coord[1],
        elbow_coord[2], 'Arm', 'x', color='red')


plt.show()
