import matplotlib.pyplot as plt
import numpy as np
import movis as mv
import json

with open("result.json") as f:
	history = json.load(f)

all_by_key = lambda x: [entry[x] for entry in history]

right_shoulder = np.array(all_by_key("right shoulder"))
left_shoulder = np.array(all_by_key("left shoulder"))

def v1():
	a = left_shoulder[:, 0]
	b = right_shoulder[:, 0]

	plt.step(np.arange(len(a)), a, label="left shoulder")
	plt.step(np.arange(len(b)), b, label="right_shoulder")
	plt.legend()
	plt.show()

def v2():
	a = left_shoulder[:, 0]
	b = right_shoulder[:, 0]

	# Сильно смягчить
	kernel = np.blackman(80)
	a = np.convolve(a, kernel, mode="same")
	b = np.convolve(b, kernel, mode="same")

	# Найти низины
	bzp = np.roll(b, 1)
	bzn = np.roll(b, -1)

	peaks = (b < bzp) * (b < bzn)
	peaks[0] = 0

	def debug():
		plt.step(np.arange(len(a)), a, label="left shoulder")
		plt.step(np.arange(len(b)), b, label="right_shoulder")
		plt.step(np.arange(len(peaks)), peaks*100, label="peaks")
		plt.legend()
		plt.show()

	return peaks

source = mv.layer.Video("test_1.mp4")
frames = int(source.duration * source.fps)

peaks = v2()
total = np.cumsum(peaks)

pushup_count_fn = lambda t: str( total[int(t*source.fps)] )

scene = mv.layer.Composition(size=(1280, 720), duration=source.duration)
scene.add_layer(source)
scene.add_layer(
	mv.layer.Text(pushup_count_fn, font_size=256, font_family='Helvetica', color='#ffffff')
)

with scene.preview(level=2):
	scene.write_video('output.mp4')
