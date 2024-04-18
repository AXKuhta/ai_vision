from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



from sklearn.manifold import TSNE

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import random_projection
from sklearn.decomposition import PCA

from gmsdb import GMSDB

import numpy as np
import cv2

# faces.data		расплющенные
# faces.images		двумерные
faces = datasets.fetch_olivetti_faces()
h, w = faces.images[0].shape

def debug():
	fig, axes = plt.subplots(1, 3)
	axes[0].imshow(faces.images[1])
	axes[1].imshow(faces.images[10])
	axes[2].imshow(faces.images[20])
	plt.show()

sift = cv2.SIFT_create(nfeatures=20)
X = faces.images
embs = []
kps = []

for i in range(X.shape[0]):
	gray = X[i, :, :, None] * 255.0
	gray = np.concatenate([gray, gray, gray], axis=-1).astype(int)
	gray = np.uint8( cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX) )

	kp = sift.detect(gray, None)

	# Иногда получается 21 keypoint
	# Из-за двух с одинаковым score?
	# В любом случае, выкинуть лишние
	kp = kp[:20]

	kps.append(kp)

	kp, desc = sift.compute(gray, kp)

	assert desc.shape[0] == 20

	for k in range(desc.shape[0]):
		embs.append(desc[k])

embs = np.array(embs)

# Правильные алгоритмы
# GMSDB
# TSNE
#
# ...но у меня работают слишком медленно
#

def clusters_v1():
	pca = PCA(n_components=2)
	z = pca.fit_transform(embs)

	model = GaussianMixture(n_components=13)
	color = model.fit_predict(embs)

	plt.scatter(*z.T, c=color)
	plt.show()


def clusters_v2():
	optimal_n = int(embs.shape[0] ** 0.5)

	pca = PCA(n_components=optimal_n)
	z = pca.fit_transform(embs)

	model = GaussianMixture(n_components=13)
	color = model.fit_predict(z)

	pca = PCA(n_components=2)
	z = pca.fit_transform(embs)

	#plt.scatter(*z.T, c=color)
	#plt.show()

	return color


def clusters_v3():
	optimal_n = int(embs.shape[0] ** 0.5)

	pca = PCA(n_components=optimal_n)
	z = pca.fit_transform(embs)

	model = AgglomerativeClustering(n_clusters=13)
	color = model.fit_predict(z)

	pca = PCA(n_components=2)
	z = pca.fit_transform(embs)

	plt.scatter(*z.T, c=color)
	plt.show()

	return color


color = clusters_v3()
imgs = []
pts_ = []

# Ищем центры кластеров
for c in np.unique(color):
	indices = np.where(color == c)[0]
	pts = embs[indices]

	centered = pts - pts.mean(0)
	dist = (centered * centered).sum(1)**0.5

	idx_local = dist.argmin()
	idx = indices[idx_local]

	im_idx = idx // 20
	kp_idx = idx % 20

	pt = kps[im_idx][kp_idx].pt

	print(f"Keypoint {idx} best represents cluster {c}")
	print(f"This is image {im_idx} kp {kp_idx}")
	print(f"Point {pt}")

	imgs.append(faces.images[im_idx])
	pts_.append(pt)

fig, axes = plt.subplots(3, 5)

for axis, img, pt in zip(axes.flatten(), imgs, pts_):
	axis.imshow(img)
	axis.scatter(*np.array(pt).T, facecolor="none", edgecolor="red", s=100**2)

plt.show()
