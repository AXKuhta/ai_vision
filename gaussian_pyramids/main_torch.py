from torch.nn.functional import conv_transpose2d, conv2d
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
import torch

# ##############################################################################
# Наш набор фильтров, 4 направленных один мылящий
# ##############################################################################
sobels = [0]*5

sobels[0]=np.array([
	[1/16,0,-1/16],
	[1/8,0,-1/8],
	[1/16,0,-1/16]
])

sobels[1]=np.array([
	[1/16,1/8,1/16],
	[0,0,0],
	[-1/16,-1/8,-1/16]
])

sobels[2]=np.array([
	[0,1/8,1/16],
	[-1/8,0,1/8],
	[-1/16,-1/8,0]
])

sobels[3]=np.array([
	[1/16,1/8,0],
	[1/8,0,-1/8],
	[0,-1/8,-1/16]
])

sobels[4]=np.array([
	[1/16,1/8,1/16],
	[1/8,1/4,1/8],
	[1/16,1/8,1/16]
])

image = plt.imread("clocks.jpg") / 255

def down(img, kernel):
	img = torch.tensor(img)
	r = img[:, :, 0]
	g = img[:, :, 1]
	b = img[:, :, 2]

	img = torch.stack([r, g, b])[None, :]

	kernel = torch.tensor([[
		kernel
	]*1]*3).double()

	blurry = conv2d(img, kernel, padding=1, stride=2, groups=3)
	r = blurry[0, 0, :, :, None]
	g = blurry[0, 1, :, :, None]
	b = blurry[0, 2, :, :, None]

	print("In", img.shape)
	print("Out", blurry.shape)

	return torch.concat([r, g, b], 2)

def up(img, kernel):
	img = torch.tensor(img)
	r = img[:, :, 0]
	g = img[:, :, 1]
	b = img[:, :, 2]

	img = torch.stack([r, g, b])[None, :]

	kernel = torch.tensor([[
		kernel
	]*1]*3).double() * 4

	blurry = conv_transpose2d(img, kernel, padding=1, output_padding=1, stride=2, groups=3)
	r = blurry[0, 0, :, :, None]
	g = blurry[0, 1, :, :, None]
	b = blurry[0, 2, :, :, None]

	print("In", img.shape)
	print("Out", blurry.shape)

	return torch.concat([r, g, b], 2)

# #################################################################################
# Т.н. вращающая пирамида
#
# Идея в том, что эти операции каким-то образом ортогональны
# На выходе получится восстановить оригинал или что-то близкое к нему, сложив всё
# #################################################################################

a = down(image, sobels[0])
b = down(image, sobels[1])
c = down(image, sobels[2])
d = down(image, sobels[3])
blur = down(image, sobels[4])

inv_a = up(a, sobels[0])
inv_b = up(b, sobels[1])
inv_c = up(c, sobels[2])
inv_d = up(d, sobels[3])
inv_blur = up(blur, sobels[4])

def disp_recover():
	recover = inv_a + inv_b + inv_c + inv_d + inv_blur

	fig, axes = plt.subplots(1, 2)
	axes[0].imshow(image[:, :, 0])
	axes[0].set_title("Input image")
	axes[1].imshow(recover[:, :, 0])
	axes[1].set_title("Recovered image")
	plt.show()

def disp_sobels():
	fig, axes = plt.subplots(1, 5)

	for i, sobel in enumerate(sobels):
		axes[i].imshow(down(image, sobel)[:, :, 0])

	plt.show()

disp_sobels()
