import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math

# Variables
dataset_path_prefix = 'data/'
csv_file = dataset_path_prefix + 'driving_log.csv'

# Correction factor for steering angle for left and right camera frames
CORRECTION_FACTOR = 0.2

BATCH_SIZE = 32
EPOCHS = 5

def normalize_input(X):
    return X / 255.0 - 0.5

def augment_data(images, labels):
    return (np.concatenate((images, np.fliplr(images))), np.concatenate((labels, labels * -1)))

def load_dataset(csv_file, path, correction_factor = 0.2):
    data = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            image_center = path + line[0]
            image_left = path + line[1].strip()
            image_right = path + line[2].strip()

            measurement = float(line[3])

            data.extend(((image_center, measurement), 
                    (image_left,  measurement + correction_factor),
                    (image_right, measurement - correction_factor)))

    # Reserve 20% of dataset for validation
    train_data, validation_data = train_test_split(data, test_size=0.2)
    return (train_data, validation_data)

def generator_routine(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size // 2):
            batch_samples = samples[offset:offset+batch_size // 2]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Load images
                image = cv2.imread(batch_sample[0])

                if image is None:
                    print('Failed to load image')
                    exit()

                # Load steering wheel angle
                measurement = float(batch_sample[1])

                images.append(image)
                measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)

            # Augment data, to avoid left turning bias
            X_train, y_train = augment_data(X_train, y_train)

            yield shuffle(X_train, y_train)

# Get tuples (img_path, steering_wheel_angle) for whole dataset with all 3 cameras (Angle is already corrected)
train_data, validation_data = load_dataset(csv_file, dataset_path_prefix, correction_factor = CORRECTION_FACTOR)

# Prepare training data generator 
train_generator = generator_routine(train_data, batch_size = BATCH_SIZE)

# Prepare validation data generator
validation_generator = generator_routine(validation_data, batch_size = BATCH_SIZE)

# Model definition
model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(4,4)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=84, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# Loss = Mean Square Error
# Optimizer = adam
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator,
                    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE),
                    validation_data = validation_generator,
                    validation_steps = math.ceil(len(validation_data) / BATCH_SIZE),
                    epochs= EPOCHS,
                    verbose = 1)

model.save('model.h5')
print("### Finished ###")