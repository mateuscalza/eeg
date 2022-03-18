import glob
import re
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

size = 512
file_lines_offset = 4
file_lines_skip = 768

classes_labels = ['espicula', 'normal', 'piscada', 'ruido']

def crawl_folder(folder):
  folder = folder.decode("utf-8")
  files = glob.glob("./" + folder + "/*/*")
  random.shuffle(files)
  for file_name in files:
    # Class
    regex = re.compile(r"./" + re.escape(folder) + r"/(\w+)/.*")
    class_label = regex.findall(file_name)[0]
    class_index = classes_labels.index(class_label)
    values = []
    # Values
    index = 0
    with open(file_name) as file:
      for line in file:
        if index >= file_lines_offset + file_lines_skip and index < file_lines_offset + file_lines_skip + size:
          string_value = line.strip()
          values.append(float(string_value))
        index += 1
    min = 0.
    max = 1.
    x = np.array(values).astype(np.float32).reshape(512,)
    x_std = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    x_scaled = x_std * (max - min) + min
    y = [
      1 if class_index == 0 else 0,
      1 if class_index == 1 else 0,
      1 if class_index == 2 else 0,
      1 if class_index == 3 else 0,
    ]
    final_y = np.array(y).astype(np.float32).reshape(4,)
    # print(x_scaled.shape)
    yield (x_scaled, final_y)

# iter = crawl_folder(b'training')
# print(next(iter)[0].shape)
# print(next(iter)[1].shape)
# print(next(iter))
# print(next(iter))
# next(iter)
# next(iter)
# next(iter)
# next(iter)
# next(iter)
# next(iter)

train_dataset = tf.data.Dataset.from_generator(
  crawl_folder,
  args = ['training'], 
  output_signature=(
    tf.TensorSpec(shape=(512,),
      dtype=tf.float32
    ),
    tf.TensorSpec(shape=(4,),
      dtype=tf.int8
    ),
  ),
  # output_types=(tf.float32, tf.float32)
)
validation_dataset = tf.data.Dataset.from_generator(
  crawl_folder,
  args = ['validation'], 
  output_signature=(
    tf.TensorSpec(shape=(512,), dtype=tf.float32),
    tf.TensorSpec(shape=(4,), dtype=tf.int8),
  )
)

# # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
# # print(train_dataset.)

model = tf.keras.Sequential([
  # layers.Conv1D(filters=64, kernel_size=4, strides = 1, activation='relu', input_shape=(8,512,1)),
  # layers.Conv1D(filters=64, kernel_size=4, strides = 1, activation='relu'),
  # layers.MaxPool1D(pool_size=2, strides=1),
  # layers.Flatten(),
  # layers.Dense(100, activation='relu'),
  # layers.Dense(4, activation = "softmax")



  # layers.Flatten(input_shape=(8, 512, 1)),
  # layers.Dense(128, activation='relu'),
  # layers.Dense(4, activation='softmax')


  # layers.Flatten(),
  layers.InputLayer(input_shape=(1, 512)),
  layers.Dense(128, activation='relu'),  
  layers.Dense(256, activation='relu'),  
  layers.Dense(256, activation='relu'),  
  layers.Dense(4, activation='softmax') 
])
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(train_dataset.batch(8), validation_data = validation_dataset.batch(8), epochs = 50)

# for element in train_dataset:
#   print(element)

# print(next(iter(train_dataset.batch(8))))

# next(iter(train_dataset.batch(8))).shape

