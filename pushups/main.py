import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import movis as mv
import json

label = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]

model = hub.load("./")
movenet = model.signatures["serving_default"]

source = mv.layer.Video("test_1.mp4")

def debug():
	frametime = list( np.linspace(0, source.duration, int(source.duration * source.fps) ) )

	for t in frametime:
		plt.clf()
		plt.imshow(source(t))
		plt.pause(.1)









def make_pred(img, keypoints_dict, label):
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img[0])
    plt.subplot(1, 3, 2)
    plt.imshow(img[0])
    plt.title('Pose')
    plt.axis('off')
    for i in range(13):
        plt.scatter(keypoints_dict[label[i]][1],keypoints_dict[label[i]][0],color='green')

    connections = [
        ('nose', 'left eye'), ('left eye', 'left ear'), ('nose', 'right eye'), ('right eye', 'right ear'),
        ('nose', 'left shoulder'), ('left shoulder', 'left elbow'), ('left elbow', 'left wrist'),
        ('nose', 'right shoulder'), ('right shoulder', 'right elbow'), ('right elbow', 'right wrist'),
        ('left shoulder', 'left hip'), ('right shoulder', 'right hip'), ('left hip', 'right hip'),
        ('left hip', 'left knee'), ('right hip', 'right knee')
    ]

    for start_key, end_key in connections:
        if start_key in keypoints_dict and end_key in keypoints_dict:
            start_point = keypoints_dict[start_key][:2]  # Take first two values
            end_point = keypoints_dict[end_key][:2]      # Take first two values
            plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], linewidth=2)

    plt.subplot(1, 3, 3)
    plt.imshow((img[0]/255)/255)
    plt.title('Only Pose Image')
    for start_key, end_key in connections:
        if start_key in keypoints_dict and end_key in keypoints_dict:
            start_point = keypoints_dict[start_key][:2]  # Take first two values
            end_point = keypoints_dict[end_key][:2]      # Take first two values
            plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], linewidth=2)
    plt.pause(.1)



frametime = list( np.linspace(0, source.duration, int(source.duration * source.fps) ) )

frames = int(source.duration * source.fps)

history = []

for i in range(frames):
	pic = source(i/frames * source.duration)
	inp = tf.image.resize_with_pad(pic[None, :, :, :3], 512, 512)
	X = tf.cast(inp, dtype=tf.int32)

	def debug():
		plt.clf()
		plt.imshow(inp[0])
		plt.pause(.1)

	outputs = movenet(X)
	keypoints = outputs['output_0'].numpy()

	max_key, key_val = keypoints[0, :, 55].argmax(), keypoints[0, :, 55].max()

	max_points = keypoints[0,max_key,:]
	max_points = max_points*512
	max_points = max_points.astype(float)

	keypoints_dict = {}
	for i in range(0,len(max_points)-5,3):
		keypoints_dict[label[i//3]] = [max_points[i],max_points[i+1],max_points[i+2]]

	history.append(keypoints_dict)

	print(keypoints_dict['right ankle'])
	print(keypoints_dict['left ankle'])

	#make_pred(X, keypoints_dict, label)

	#X = tf.cast(inp, dtype=tf.int32)
	#print(X)

with open("result.json", "w") as f:
	json.dump(history, f)

#debug()
