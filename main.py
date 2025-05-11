import matplotlib.pyplot as plt
import numpy as np


spiral_data = np.loadtxt('./Spiral3d.csv', 'double', delimiter=',')
aero_data = np.loadtxt('./aerogerador.dat', 'float', delimiter='\t')

fig = plt.figure()
ax_spiral = fig.add_subplot(211, projection='3d')
ax_spiral.scatter(spiral_data[:, 0], spiral_data[:, 1], spiral_data[:, 2], c=spiral_data[:, 3], cmap='viridis')
ax_spiral.set_xlabel('X')
ax_spiral.set_ylabel('Y')
ax_spiral.set_zlabel('Z')

ax_aero = fig.add_subplot(212)
ax_aero.plot(aero_data[:, 0], aero_data[:, 1])


plt.tight_layout()
plt.show()
