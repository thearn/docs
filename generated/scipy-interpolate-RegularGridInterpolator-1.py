# Evaluate a simple example function on the points of a 3D grid:

import numpy as np
from scipy.interpolate import RegularGridInterpolator
def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z
x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))

# ``data`` is now a 3D array with ``data[i,j,k] = f(x[i], y[j], z[k])``.
# Next, define an interpolating function from this data:

my_interpolating_function = RegularGridInterpolator((x, y, z), data)

# Evaluate the interpolating function at the two points
# ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
my_interpolating_function(pts)
# array([ 125.80469388,  146.30069388])

# which is indeed a close approximation to
# ``[f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)]``.

# With the spline interpolation methods it is possible to compute smooth
# gradients for a variety of purposes, such as numerical optimization.

# To demonstrate this, let's define a function with known gradients for
# demonstration, and create grid sample axes with a variety of sizes:

from scipy.optimize import fmin_bfgs
def F(u, v, z, w):
    return (u - 5.234)**2 + (v - 2.128)**2 + (z - 5.531)**2 + (w - 0.574)**2
def dF(u, v, z, w):
    return 2 * (u - 5.234), 2 * (v - 2.128), 2 * (z - 5.531), 2 * (w - 0.574)
np.random.seed(0)
U = np.linspace(0, 10, 10)
V = np.random.uniform(0, 10, 10)
Z = np.random.uniform(0, 10, 10)
W = np.linspace(0, 10, 10)
V.sort(), Z.sort()
# (None, None)
points = [U, V, Z, W]
values = F(*np.meshgrid(*points, indexing='ij'))

# Now, define a random sampling point

x = np.random.uniform(1, 9, 4)

# With the cubic interpolation method, gradient information will be
# available:

interp = RegularGridInterpolator(
    points, values, method="cubic", bounds_error=False, fill_value=None)

# This provides smooth interpolation values for approximating the original
# function and its gradient:

F(*x), interp(x)
# (85.842906385928046, array(85.84290638592806))
dF(*x)
# (7.1898934757242223, 10.530537027467577, -1.6783302039530898, 13.340466820583288)
interp.gradient(x)
# array([  7.18989348,  10.53053703,  -1.6783302 ,  13.34046682])

# The `gradient` method can conveniently be passed as an argument to any
# procedure that requires gradient information, such as
# ``scipy.optimize.fmin_bfgs``:

opt = fmin_bfgs(interp, x, fprime=interp.gradient)
# Optimization terminated successfully.
# Current function value: 0.000000
# Iterations: 3
# Function evaluations: 5
# Gradient evaluations: 5

# Despite the course data grid and non-homogeneous axis dimensions, the
# computed minimum matches the known solution very well:

print(opt)
# [ 5.234  2.128  5.531  0.574]

# All available interpolation methods can be compared based on the task
# of fitting a course sampling to interpolate a finer representation:

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)
# >>>
def F(u, v):
    return u * np.cos(u * v) + v * np.sin(u * v)
fit_points = [np.linspace(0, 3, 8), np.linspace(0, 3, 8)]
values = F(*np.meshgrid(*fit_points, indexing='ij'))
test_points = [np.linspace(fit_points[0][0], fit_points[0][-1], 80), np.linspace(
    fit_points[1][0], fit_points[1][-1], 80)]
ut, vt = np.meshgrid(*test_points, indexing='ij')
true_values = F(ut, vt)
pts = np.array([ut.ravel(), vt.ravel()]).T
plt.figure()
for i, method in enumerate(RGI.methods()):
    plt.subplot(2, 3, i + 1)
    interp = RGI(fit_points, values, method=method)
    im = interp(pts).reshape(80, 80)
    plt.imshow(im, interpolation='nearest')
    plt.gca().axis('off')
    plt.title(method)
plt.subplot(2, 3, 6)
plt.title("True values")
plt.gca().axis('off')
plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout()
plt.imshow(true_values, interpolation='nearest')
plt.show()

# As expected, the cubic and quintic spline interpolations are closer to the
# true values, though are more expensive to compute than with linear or
# nearest. The slinear interpolation also matches the linear interpolation.

# The computed gradient fields can also be visualized:

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (6, 3)
n = 30
fit_points = [np.linspace(0, 3, 8), np.linspace(0, 3, 8)]
values = F(*np.meshgrid(*fit_points, indexing='ij'))
test_points = [np.linspace(0, 3, n), np.linspace(0, 3, n)]
ut, vt = np.meshgrid(*test_points, indexing='ij')
true_values = F(ut, vt)
pts = np.array([ut.ravel(), vt.ravel()]).T
interp = RGI(fit_points, values, method='cubic')
im = interp(pts).reshape(n, n)
gradient = interp.gradient(pts).reshape(n, n, 2)
plt.figure()
plt.subplot(121)
plt.title("cubic fit")
plt.imshow(im[::-1], interpolation='nearest')
plt.gca().axis('off')
plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout()
plt.subplot(122)
plt.title('gradients')
plt.gca().axis('off')
plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout()
plt.quiver(ut, vt, gradient[:, :, 0], gradient[:, :, 1], width=0.01)
plt.show()

# Higher-dimensional gradient field predictions and visualizations can be
# done the same way:

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import rcParams
rcParams['figure.figsize'] = (7, 6)
# set up 4D test problem
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
n = 10
pts = [np.linspace(-1, 1, n), np.linspace(-1, 1, n), np.linspace(-1, 1, n)]
x, y, z = np.meshgrid(*pts, indexing='ij')
voxels = np.array([x.ravel(), y.ravel(), z.ravel()]).T
values = np.sin(x) * y**2 - np.cos(z)
# interpolate the created 4D data
interp = RegularGridInterpolator(pts, values, method='cubic')
gradient = interp.gradient(voxels).reshape(n, n, n, 3)
u, v, w = gradient[:, :, :, 0], gradient[:, :, :, 1], gradient[:, :, :, 2]
# Plot the predicted gradient field
fig.tight_layout()
ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
plt.show()
