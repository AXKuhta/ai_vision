import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse

from skimage.feature import canny
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

# Line finding using the Probabilistic Hough Transform
image = plt.imread("horizon.png")[:, :, :3]
saturation = rgb2hsv(image)[:, :, 1]
edges = canny(saturation, 0, 0.25, 1)

def debug_edges():
	plt.imshow(edges)
	plt.show()

from skimage.transform import hough_line, hough_line_peaks

# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(edges, theta=tested_angles)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(edges, cmap="gray")
ax[0].set_title('Input edges')
ax[0].set_axis_off()

angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(d).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          d[-1] + d_step, d[0] - d_step]

ax[1].imshow(np.log(1 + h), extent=bounds, cmap="gray", aspect=1 / 1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

def desmos_print(slope, x0, y0):
    print(f"{-slope:.3f}x + {(x0*slope - y0):.3f}")
    print(" ============= ")

lines = []

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    if np.abs(angle) < np.deg2rad(20):
        color="gray"
    else:
        color="blue"

    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    slope = np.tan(angle + np.pi/2)

    ax[2].axline((x0, y0), slope=slope, c=color)

    if np.abs(angle) > np.deg2rad(20):
    	lines.append( [-slope, x0*slope - y0] )

def intersect_lines(line1, line2):
	a, b = line1
	u, v = line2

	fac = a - u
	val = -b + v

	x = val / fac
	y = a*x + b

	return (x, y)

xlist = []
ylist = []

for a, b in combinations(lines, 2):
	x, y = intersect_lines(a, b)
	xlist.append(x)
	ylist.append(-y)

ax[2].scatter(xlist, ylist)
plt.tight_layout()
plt.show()
