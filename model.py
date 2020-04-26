import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Prepare lists in which images and labels will be placed
images = []
measurements = []

for line in lines:
    # 1st part of line is center image
    image_path = line[0]
    image = cv2.imread('data/' + image_path)
    images.append(np.array(image))

    # 4th part of line is steering wheel angle
    measurement = float(line[3])
    measurements.append(measurement)

# Images and measurements are now loaded
X_train = np.array(images)
Y_train = np.array(measurements)

print(X_train.shape)
print(Y_train.shape)

model = Sequential()
model.add(Flatten(input_shape = (160, 320, 3)))
model.add(Dense(1))

# Loss = Mean Square Error
# Optimizer = adam
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save('model.h5')