import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 100)

fig = plt.figure()

# https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
# add_subplot(*args, **kwargs)
# add_subplot(nrows, ncols, index, **kwargs)
# add_subplot(pos, **kwargs)
# add_subplot(ax)

ax1 = fig.add_subplot(221)
ax1.plot(x, x)

ax2 = fig.add_subplot(222)
ax2.plot(x, -x)

ax3 = fig.add_subplot(223)
ax3.plot(x, x ** 2)

ax4 = fig.add_subplot(224)
ax4.plot(x, np.log(x))

plt.show()
