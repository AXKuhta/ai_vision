from skimage.transform import warp, ProjectiveTransform
import matplotlib.pyplot as plt
import numpy as np

def display(source, points, section):
	fig, axes = plt.subplots(1, 2)

	axes[0].set_title("Source image")
	axes[0].imshow(source)
	axes[0].scatter(*points.T, c="red")
	axes[1].imshow(section)
	axes[1].set_title("Section")

	plt.show()

def tiles_1():
	arr = plt.imread("floor1.png")
	mat = ProjectiveTransform()

	before = np.array([
		[254, 120], [542, 141],
		[116, 234], [540, 285]
	])

	after = np.array([
		[0, 0], [200, 0],
		[0, 200], [200, 200]
	])


	mat.estimate(before, after)
	print(mat.params)

	recover = warp(arr, mat.inverse, output_shape=[200, 200])

	display(arr, before, recover)

def tiles_2():
	arr = plt.imread("floor2.png")
	mat = ProjectiveTransform()

	before = np.array([
		[165, 143], [412, 135],
		[142, 407], [461, 387]
	])

	after = np.array([
		[0, 0], [200, 0],
		[0, 200], [100, 200]
	]) + np.array([100, 0])

	mat.estimate(before, after)
	print(mat.params)

	recover = warp(arr, mat.inverse, output_shape=[200, 300])

	display(arr, before, recover)

tiles_1()
tiles_2()
