import numpy as np
import matplotlib.pyplot as plt
 
# Create data
N = 500
x = np.random.rand(N)
y = np.random.rand(N)
colors = (0,0,0)
area = np.pi*3

# matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None,
# norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None,
# edgecolors=None, *, data=None, **kwargs)

# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
