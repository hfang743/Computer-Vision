from __main__ import *
from imageCorrection import *
import matplotlib.pyplot as plt

l = openVar('Allvisited points')
l = np.array(l)
plt.scatter(l[:, 0], l[:, 1], color='r')
plt.scatter(l[0, 0], l[0, 1], color='g')
plt.scatter(l[len(l) - 1, 0], l[len(l) - 1, 1], color='b')

plt.show()
