import numpy
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


a = [-0.3772191, 0.6764629, 2.477132, -0.3672488, 0.4930182, 2.566909, -0.529421, 0.3558396, 2.57627, -0.1943015, 0.4311023, 2.561145, -0.5675833, 0.08756277, 2.615399, -0.03334243, 0.5791205, 2.433389, -0.5894274, -0.1222883, 2.591342, -0.1127574, 0.7756686, 2.344967, -0.5723279, -0.2112776, 2.548121, -0.1788912, 0.8131377, 2.315454, -0.3496525, 0.1472809, 2.577553, -0.3463077, 0.0782887, 2.574845, -0.419842, -0.005263386, 2.57652, -0.2643511, 0.007698199, 2.571115, -0.4584759, -0.4603404, 2.673417, -0.2359848, -0.4796088, 2.666875, -0.4610319, -0.8478795, 2.715237, -0.2295792, -0.8651484, 2.683695, -0.4660459, -0.8985245, 2.631478, -0.2079941, -0.9369207, 2.647322]

print(a)
b = numpy.reshape(a, (20,3))
print(b)

fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

for points in b:
    ax.scatter(points[0], points[1], points[2])

#ax.scatter(2,3,4) # plot the point (2,3,4) on the figure

plt.show()