#  /* Copyright 2018 kunming.xie
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *    http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#   */

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
