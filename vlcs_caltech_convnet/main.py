import matplotlib.pyplot as plt
from glob import glob
import numpy as np

import tensorflow as tf
import cv2
import gc

class DataLoader:
	def __init__(self, datasettype='train'):
		if datasettype == "test":
			self.filelist = glob("VLCS/CALTECH/test/*/*")
		elif datasettype == "train":
			self.filelist = glob("VLCS/CALTECH/train/*/*")
			self.filelist = self.filelist[:len(self.filelist)*9//10]
		elif datasettype == "val":
			self.filelist = glob("VLCS/CALTECH/train/*/*")
			self.filelist = self.filelist[len(self.filelist)*9//10:]
		else:
			raise Exception("Unk datasettype")

		np.random.shuffle(self.filelist)

	def __call__(self):
		for fname in self.filelist:
			img = cv2.imread(fname, cv2.IMREAD_COLOR) / 255.0
			label = int( fname.split("/")[-2] )

			yield img, tf.keras.utils.to_categorical(label, 5)


train_loader = DataLoader(datasettype="train")
train_dataset = tf.data.Dataset.from_generator(
	train_loader,
	output_types=(tf.float32, tf.int32),
	output_shapes=(tf.TensorShape([227, 227, 3]), tf.TensorShape(5))
)

val_loader = DataLoader(datasettype="val")
val_dataset = tf.data.Dataset.from_generator(
	val_loader,
	output_types=(tf.float32, tf.int32),
	output_shapes=(tf.TensorShape([227, 227, 3]), tf.TensorShape(5))
)

test_loader = DataLoader(datasettype="test")
test_dataset = tf.data.Dataset.from_generator(
	test_loader,
	output_types=(tf.float32, tf.int32),
	output_shapes=(tf.TensorShape([227, 227, 3]), tf.TensorShape(5))
)

train_batches = train_dataset.batch(16)
val_batches = val_dataset.batch(16)
test_batches = test_dataset.batch(16)

model = tf.keras.Sequential([
	tf.keras.layers.InputLayer([227, 227, 3]),
	tf.keras.layers.Conv2D(5, 3, strides=2, padding="same", activation="relu"),
	tf.keras.layers.Conv2D(5, 3, strides=2, padding="same", activation="relu"),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(5, activation="softmax")
])

model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_batches, epochs=3, validation_data=val_batches)
model.evaluate(test_batches)
