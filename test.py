import glob
import re
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

file_name = "./testing/espicula/Esp¡culaOnda_PacC3_F4-Pz_000343_2048_148ms.pdr"

size = 512
file_lines_offset = 4
file_lines_skip = 768

values = []

index = 0
with open(file_name) as file:
  for line in file:
    if index >= file_lines_offset + file_lines_skip and index < file_lines_offset + file_lines_skip + size:
      string_value = line.strip()
      values.append(float(string_value))
    index += 1

min = 0.
max = 1.

initial_x = np.array(values).astype(np.float32)
x_std = (initial_x - initial_x.min(axis=0)) / (initial_x.max(axis=0) - initial_x.min(axis=0))
x_scaled = x_std * (max - min) + min
x_test = np.array([x_scaled])

print(x_test)

model = load_model('./model/model.h5')
result = model.predict(x_test)

classes = range(4)
classes_labels_names = ['Espícula', 'Normal', 'Piscada', 'Ruído']
for index in classes:
  print(classes_labels_names[index], '= %.2f%%' % (result[0][index] * 100.0))

