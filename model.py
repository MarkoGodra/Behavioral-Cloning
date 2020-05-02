import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D, ELU
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
from datetime import datetime

# Correction factor for steering angle for left and right camera frames
CORRECTION_FACTOR = 0.2
BATCH_SIZE = 128
EPOCHS = 3

# Normalization
def normalize_input(X):
    return X / 255.0 - 0.5

# Data augmentation, flips all data w.r.t. y axis
def augment_data(images, labels):
    return (np.concatenate((images, np.fliplr(images))), np.concatenate((labels, labels * -1)))

# Loads dataset
# Returns -> list of (image_path, meassurement)
def load_dataset(csv_file, path, correction_factor = 0.2):
    # Prepare data list
    data = []

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        
        # For each line in csv file
        for line in reader:

            # Extract names of each ceneter, left and right image
            image_center = path + line[0]
            image_left = path + line[1].strip()
            image_right = path + line[2].strip()

            # Extract value of steering wheel angle
            measurement = float(line[3])

            # Append data with angle correction for left and right images
            data.extend(((image_center, measurement), 
                    (image_left,  measurement + correction_factor),
                    (image_right, measurement - correction_factor)))

    # Reserve 20% of dataset for validation
    train_data, validation_data = train_test_split(data, test_size=0.2)
    return (train_data, validation_data)

# Generator routine - Used for RAM usage reduction, since
# we can't load all images at once
def generator_routine(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Neverending loop

        # Shuffle data samples
        shuffle(samples)

        # For each batch
        for offset in range(0, num_samples, batch_size):

            # Cut of batch of data
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            # For each (image_path, meassurement) tuple
            for batch_sample in batch_samples:

                # Load the image
                image = cv2.imread(batch_sample[0])

                # Check if image is read correctly
                if image is None:
                    print('Failed to load image:', batch_sample)
                    exit()

                # Load steering wheel angle
                measurement = float(batch_sample[1])

                # Append image and measurements for network input and label
                images.append(image)
                measurements.append(measurement)

            # Convert data to np.array since this is what keras expect
            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)

# Get tuples (img_path, steering_wheel_angle) for whole dataset with all 3 cameras (Angle is already corrected)
udacity_data_train, udacity_data_validation = load_dataset('data/driving_log.csv', 'data/', correction_factor = CORRECTION_FACTOR)
train_data, validation_data = load_dataset('my-data/driving_log.csv', 'my-data/', correction_factor = CORRECTION_FACTOR)

# Combine recovery data with given udacity data
train_data = train_data + udacity_data_train
validation_data = validation_data + udacity_data_validation

# Shuffle newly generated datasets
train_data = shuffle(train_data)
validation_data = shuffle(validation_data)

# Prepare training data generators 
train_generator = generator_routine(train_data, batch_size = BATCH_SIZE)
validation_generator = generator_routine(validation_data, batch_size = BATCH_SIZE)

# Model definition
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping = ((75,20), (0,0))))
model.add(Conv2D(24, (5, 5), subsample = (2, 2), activation = 'relu'))
model.add(Conv2D(36, (5, 5), subsample = (2, 2), activation = 'relu'))
model.add(Conv2D(48, (5, 5), subsample = (2, 2), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))

# Loss = Mean Square Error
# Optimizer = adam
model.compile(loss = 'mse', optimizer = 'adam')
fitting_history = model.fit_generator(train_generator,
                    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE),
                    validation_data = validation_generator,
                    validation_steps = math.ceil(len(validation_data) / BATCH_SIZE),
                    epochs= EPOCHS,
                    verbose = 1)

# Save the parameters for newly formed model
model.save('model.h5')

# Plot loss data
plt.plot(fitting_history.history['loss'])
plt.plot(fitting_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

# Save plot
plt.savefig('images/training_loss_' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))