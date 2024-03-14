from tensorflow.nn import conv2d_transpose, conv2d
import matplotlib.pyplot as plt
from skimage import data
import numpy as np

# #################################################################################
# Набор фильтров
# #################################################################################
kernels = [0]*5

kernels[0]=np.array([
	[1/16,0,-1/16],
	[1/8,0,-1/8],
	[1/16,0,-1/16]
])

kernels[1]=np.array([
	[1/16,1/8,1/16],
	[0,0,0],
	[-1/16,-1/8,-1/16]
])

kernels[2]=np.array([
	[0,1/8,1/16],
	[-1/8,0,1/8],
	[-1/16,-1/8,0]
])

kernels[3]=np.array([
	[1/16,1/8,0],
	[1/8,0,-1/8],
	[0,-1/8,-1/16]
])

kernels[4]=np.array([
	[1/16,1/8,1/16],
	[1/8,1/4,1/8],
	[1/16,1/8,1/16]
])

def down(img, kernel):
	r = img[:, :, 0][None, :, :, None]
	g = img[:, :, 1][None, :, :, None]
	b = img[:, :, 2][None, :, :, None]

	kernel = kernel[:, :, None, None]

	r = conv2d(r, kernel, padding="SAME", strides=2)[0]
	g = conv2d(g, kernel, padding="SAME", strides=2)[0]
	b = conv2d(b, kernel, padding="SAME", strides=2)[0]

	return np.concatenate([r, g, b], 2)

def up(img, kernel):
	r = img[:, :, 0][None, :, :, None]
	g = img[:, :, 1][None, :, :, None]
	b = img[:, :, 2][None, :, :, None]
	h, w, c = img.shape
	h *= 2
	w *= 2

	kernel = kernel[:, :, None, None]

	r = conv2d_transpose(r, kernel, output_shape=[h, w], padding="SAME", strides=2)[0]
	g = conv2d_transpose(g, kernel, output_shape=[h, w], padding="SAME", strides=2)[0]
	b = conv2d_transpose(b, kernel, output_shape=[h, w], padding="SAME", strides=2)[0]

	return np.concatenate([r, g, b], 2)

# #################################################################################
# Пирамида
# #################################################################################

image = plt.imread("clocks.jpg") / 255

a = down(image, kernels[0])
b = down(image, kernels[1])
c = down(image, kernels[2])
d = down(image, kernels[3])
blur = down(image, kernels[4])

inv_a = up(a, kernels[0])
inv_b = up(b, kernels[1])
inv_c = up(c, kernels[2])
inv_d = up(d, kernels[3])
inv_blur = up(blur, kernels[4])

def disp_recover():
	recover = inv_a + inv_b + inv_c + inv_d + inv_blur
	recover *= 4

	fig, axes = plt.subplots(1, 2)
	axes[0].imshow(image)
	axes[0].set_title("Input image")
	axes[1].imshow(recover)
	axes[1].set_title("Recovered image")
	plt.show()

def disp_kernels():
	fig, axes = plt.subplots(1, 5)

	for i, kernel in enumerate(kernels):
		axes[i].imshow(down(image, kernel)[:, :, 0])
		axes[i].set_title(f"Kernel {i}")

	plt.show()

disp_recover()
disp_kernels()
