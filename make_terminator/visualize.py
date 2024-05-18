from skimage.transform import warp, ProjectiveTransform, SimilarityTransform
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import movis as mv
import json

with open("result.json") as f:
	history = json.load(f)

all_by_key = lambda x: [entry[x] for entry in history]

left_shoulder = np.array(all_by_key("left shoulder"))
left_elbow = np.array(all_by_key("left elbow"))
left_wrist = np.array(all_by_key("left wrist"))

right_hip = np.array(all_by_key("right hip"))
right_knee = np.array(all_by_key("right knee"))
right_ankle = np.array(all_by_key("right ankle"))

source = mv.layer.Video("test.mp4")
frames = int(source.duration * source.fps)

# Вырезаем представляющие интерес регионы терминатора
terminator = plt.imread("terminator.jpg").copy()
terminator[terminator >= np.ones(3)*252] = 0

rect = terminator.copy()
rect[132:290, 451:518] = 0
upper_arm = terminator - rect

rect = terminator.copy()
rect[290:457, 451:518] = 0
lower_arm = terminator - rect

rect = terminator.copy()
rect[370:530, 276:317] = 0
upper_leg = terminator - rect

rect = terminator.copy()
rect[535:696, 265:303] = 0
lower_leg = terminator - rect

# Точки терминатора
ref_s = [463, 153] # Плечо
ref_e = [478, 277] # Локоть
ref_w = [491, 381] # Кисть

"""
ref_h = [369, 320] # Таз
ref_k = [529, 290] # Колено
ref_a = [692, 283] # Лодыжка
"""

ref_h = [320, 369] # Таз
ref_k = [290, 529] # Колено
ref_a = [283, 692] # Лодыжка


plt.imshow(terminator)
plt.show()

# Базовое видео
def overlay_video_fn(t):
	frame = source(t)
	frame = tf.image.resize_with_pad(frame, 512, 512).numpy()

	return np.uint8(frame)

# Наложение верхней части руки
def overlay_upper_arm_fn(t):
	idx = int(t * source.fps)

	s = left_shoulder[idx][:2][::-1]
	e = left_elbow[idx][:2][::-1]

	before = np.array([ref_s, ref_e])
	after = np.array([s, e])

	mat = SimilarityTransform()
	mat.estimate(before, after)

	color = warp(upper_arm, mat.inverse, output_shape=[512, 512])
	alpha = color[:, :, 0] > 0

	rgba = np.dstack( [color, alpha] ) * 255

	return np.uint8(rgba)

# Наложение нижней части руки
def overlay_lower_arm_fn(t):
	idx = int(t * source.fps)

	e = left_elbow[idx][:2][::-1]
	w = left_wrist[idx][:2][::-1]

	before = np.array([ref_e, ref_w])
	after = np.array([e, w])

	mat = SimilarityTransform()
	mat.estimate(before, after)

	color = warp(lower_arm, mat.inverse, output_shape=[512, 512])
	alpha = color[:, :, 0] > 0

	rgba = np.dstack( [color, alpha] ) * 255

	return np.uint8(rgba)


# Наложение верхней части ноги
def overlay_upper_leg_fn(t):
	idx = int(t * source.fps)

	h = right_hip[idx][:2][::-1]
	k = right_knee[idx][:2][::-1]

	before = np.array([ref_h, ref_k])
	after = np.array([h, k])

	mat = SimilarityTransform()
	mat.estimate(before, after)

	color = warp(upper_leg, mat.inverse, output_shape=[512, 512])
	alpha = color[:, :, 0] > 0

	rgba = np.dstack( [color, alpha] ) * 255

	return np.uint8(rgba)

# Наложение нижней части ноги
def overlay_lower_leg_fn(t):
	idx = int(t * source.fps)

	k = right_knee[idx][:2][::-1]
	a = right_ankle[idx][:2][::-1]

	before = np.array([ref_k, ref_a])
	after = np.array([k, a])

	mat = SimilarityTransform()
	mat.estimate(before, after)

	color = warp(lower_leg, mat.inverse, output_shape=[512, 512])
	alpha = color[:, :, 0] > 0

	rgba = np.dstack( [color, alpha] ) * 255

	return np.uint8(rgba)


scene = mv.layer.Composition(size=(512, 512), duration=source.duration)
scene.add_layer(overlay_video_fn)
scene.add_layer(overlay_upper_arm_fn)
scene.add_layer(overlay_lower_arm_fn)
scene.add_layer(overlay_upper_leg_fn)
scene.add_layer(overlay_lower_leg_fn)

#with scene.preview(level=2):

scene.write_video('output.mp4')
