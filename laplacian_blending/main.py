from tensorflow.nn import conv2d_transpose, conv2d
import matplotlib.pyplot as plt
from skimage import data
import numpy as np

gaussian = np.array([
	[1/16,1/8,1/16],
	[1/8,1/4,1/8],
	[1/16,1/8,1/16]
])

def down(img):
	r = img[:, :, 0][None, :, :, None]
	g = img[:, :, 1][None, :, :, None]
	b = img[:, :, 2][None, :, :, None]

	kernel = gaussian[:, :, None, None]

	r = conv2d(r, kernel, padding="SAME", strides=2)[0]
	g = conv2d(g, kernel, padding="SAME", strides=2)[0]
	b = conv2d(b, kernel, padding="SAME", strides=2)[0]

	return np.concatenate([r, g, b], 2)

def up(img):
	r = img[:, :, 0][None, :, :, None]
	g = img[:, :, 1][None, :, :, None]
	b = img[:, :, 2][None, :, :, None]
	h, w, c = img.shape
	h *= 2
	w *= 2

	kernel = gaussian[:, :, None, None]

	r = conv2d_transpose(r, kernel, output_shape=[h, w], padding="SAME", strides=2)[0]
	g = conv2d_transpose(g, kernel, output_shape=[h, w], padding="SAME", strides=2)[0]
	b = conv2d_transpose(b, kernel, output_shape=[h, w], padding="SAME", strides=2)[0]

	return np.concatenate([r, g, b], 2)*4

# #################################################################################
# Пирамида
# #################################################################################

def encode(image):
	g = []
	l = []

	for i in range(3):
		lowres = down(image)
		delta = image - up(lowres)

		g.append(lowres)
		l.append(delta)

		image = lowres

	return g, l

head = plt.imread("billy-herrington.jpg") / 255
terminator = plt.imread("terminator.jpg") / 255

mask = np.zeros([512, 512, 3])
mask[:, :256, :] = 1.0

#image *= (1 - np.kaiser(512, 50)[:, None])

mask_g, mask_l = encode(mask)
head_g, head_l = encode(head)
term_g, term_l = encode(terminator)

m = mask_g[2]
g = term_g[2]*(1 - m) + head_g[2]*m

m = mask_g[1]
l = term_l[2]*(1 - m) + head_l[2]*m

g = up(g) + l

m = mask_g[0]
l = term_l[1]*(1 - m) + head_l[1]*m

g = up(g) + l

m = mask
l = term_l[0]*(1 - m) + head_l[0]*m

g = up(g) + l

plt.imshow(g)
plt.show()

"""
fig, axes = plt.subplots(1, 4)
axes[0].imshow(image)
axes[1].imshow(g1)
axes[2].imshow(g2)
axes[3].imshow(g3)
plt.tight_layout()
plt.show()
"""

"""
d0 = up(g3) + l3
d1 = up(d0) + l2
d2 = up(d1) + l1

fig, axes = plt.subplots(1, 4)
axes[0].imshow(g3)
axes[1].imshow(d0)
axes[2].imshow(d1)
axes[3].imshow(d2)
plt.tight_layout()
plt.show()
"""
