import glob
import re
import random
import shutil
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

size = 512
file_lines_offset = 4
file_lines_skip = 768

classes_labels = ['espicula', 'normal', 'piscada', 'ruido']

random.seed(1)
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
    yield (x_scaled, final_y)

train_dataset = tf.data.Dataset.from_generator(
  crawl_folder,
  args = ['training'], 
  output_signature=(
    tf.TensorSpec(shape=(512,), dtype=tf.float32),
    tf.TensorSpec(shape=(4,), dtype=tf.float32),
  ),
)
validation_dataset = tf.data.Dataset.from_generator(
  crawl_folder,
  args = ['validation'], 
  output_signature=(
    tf.TensorSpec(shape=(512,), dtype=tf.float32),
    tf.TensorSpec(shape=(4,), dtype=tf.float32),
  )
)

batch_size = 1
if os.path.exists("logs"):
  shutil.rmtree("./logs")

model = tf.keras.Sequential([
  layers.Dense(128, activation='relu'),
  layers.Dense(1024, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(4, activation='softmax')
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1, profile_batch=0)

# Compile
model.compile(
  optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.001,
    name='SGD'
  ),
  loss = "categorical_crossentropy",
  metrics = ["accuracy"])

# Train
model.fit(train_dataset.batch(batch_size),
  validation_data = validation_dataset.batch(batch_size),
  epochs = 100,
  callbacks = [tensorboard_callback]
)

# Save model
model.save('model/model.h5')

# Evaluate from training
_, acc = model.evaluate(train_dataset.batch(batch_size), verbose=0)
print('Precis??o (treinamento) = %.2f%%' % (acc * 100.0))

# Evaluate from validation
_, acc = model.evaluate(validation_dataset.batch(batch_size), verbose=0)
print('Precis??o (valida????o) = %.2f%%' % (acc * 100.0))
