from skimage.transform import warp, ProjectiveTransform
import matplotlib.pyplot as plt
import numpy as np

image = plt.imread("cube.jpg") / 255
texture = plt.imread("texture3.png")

def apply_texture(image, texture, before, after):
	mat = ProjectiveTransform()

	mat.estimate(before, after)
	print(mat.params)

	mask = 1.0 - warp(np.ones([512, 512, 3]), mat.inverse, output_shape=image.shape)
	recover = warp(texture, mat.inverse, output_shape=image.shape)

	image *= mask
	image += recover

	return image

p1 = [1349, 114]
p2 = [2656, 626]
p3 = [687, 611]
p4 = [2058, 1168]
p5 = [852, 1320]
p6 = [2123, 1941]
p7 = [2673, 1337]

before = np.array([
	[0, 0], [511, 0],
	[0, 511], [511, 511]
])

after_a = np.array([p4, p2, p6, p7])
after_b = np.array([p3, p4, p5, p6])
after_c = np.array([p1, p2, p3, p4])

# Добавлено немного затемнения, с ним выглядит лучше
image = apply_texture(image, texture * 0.60, before, after_a)
image = apply_texture(image, texture * 0.75, before, after_b)
image = apply_texture(image, texture, before, after_c)

plt.imshow(image)
plt.show()
