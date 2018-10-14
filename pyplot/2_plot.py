# matplotlib.pyplot is a collection of command style functions that make
# matplotlib work like MATLAB.

# https://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt

# API
# https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
#
# matplotlib.pyplot.plot(*args, **kwargs)
#
# Plot lines and/or markers to the Axes. args is a variable length argument,
# allowing for multiple x, y pairs with an optional format string. For example,
# each of the following is legal:

# You may be wondering why the x-axis ranges from 0-3 and the y-axis from 1-4.
# If you provide a single list or array to the plot() command, matplotlib
# assumes it is a sequence of y values, and automatically generates the x values
# for you. Since python ranges start with 0, the default x vector has the same
# length as y but starts with 0. Hence the x data are [0,1,2,3].
plt.plot([1, 2, 3, 4])

plt.show()
