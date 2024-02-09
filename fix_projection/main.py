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

def photo():
	arr = plt.imread("small.jpg")
	mat = ProjectiveTransform()

	before = np.array([
		[2607, 274], [3806, 1147],
		[821, 817], [2017, 2240]
	]) / 3.575

	after = np.array([
		[0, 0], [200, 0],
		[0, 200], [200, 200]
	])


	mat.estimate(before, after)
	print(mat.params)

	recover = warp(arr, mat.inverse, output_shape=[200, 200])

	display(arr, before, recover)

photo()
