import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D, ELU
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
from datetime import datetime
from random import randrange
from random import randint
import pandas as pd
import os
from imgaug import augmenters as iaa

# Tunable params
CORRECTION_FACTOR = 0.2
BATCH_SIZE = 100
EPOCHS = 20
DROPOUT_RATE = 0.30
SINGLE_ANGLE_MAX_COUNT = 400

data_dir = 'data-reverse-recovery'

# Random zoom
def zoom(image):
    # Affine translations involve translation, scaling, rotation, shear
    zoom = iaa.Affine(scale=(1, 1.3))
    
    # Apply augmentation - slight zoom
    image = zoom.augment_image(image)

    return image

# Random translation
def translate_image(image):
    # Affine translations involve translation, scaling, rotation, shear
    translate = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})

    # Apply augmentation - translation
    image = translate.augment_image(image)

    return image

# Random brightness
def change_brightness(image):
    # Multiply image with random value from specified range
    brightness = iaa.Multiply((0.25, 1.15))

    # Apply brightness augmentation
    image = brightness.augment_image(image)

    return image

# Image flipping
def flip_image(image, steering_angle):
    # Use cv2 for this
    image = cv2.flip(image,1)

    # Flip steering angle
    steering_angle = -steering_angle

    return image, steering_angle

# This function augments image in 4 different ways
# Each augmentation has 50% chance to be applied
def random_augment(image, steering_angle):
    # Do coin flip
    if coin_flip(2) == True:
        image = translate_image(image)

    # Do coin flip
    if coin_flip(2) == True:
        image = zoom(image)

    # Do coin flip
    if coin_flip(2) == True:
        # Random brightness
        image = change_brightness(image)

    # Do coin flip
    if coin_flip(2) == True:
        # Flip image
        image, steering_angle = flip_image(image, steering_angle)
    
    return image, steering_angle

# Normalization
def normalize_input(X):
    return X / 255.0 - 0.5

# Helper function that returns true in 1/rate probability
def coin_flip(rate):
    return randrange(rate) == 0

# Preprocess image
def preprocess_image(image):
    # Crop image to remove sky and hood of car
    # Isolate ROI
    image = image[60:135,:,:]

    # Convert to YUV color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Apply blur to supress noise
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Resize image to nvidia model native size (66, 200, 3)
    image = cv2.resize(image, (200, 66))

    # Normalize the image
    image = image/255

    return image
    
# Load data into panda dictionary
def load_data(data_dir):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names = columns)
    return data

# Helper function for acquiring histogram of steering angles
def create_histogram(data, number_of_bins = 35):
    hist, bins = np.histogram(data['steering'], number_of_bins)
    return hist, bins

# Displays histogram of steering angles for given dataset
def display_histogram(data, number_of_hist_bins = 35):
    
    # Create histogram for steering angle
    hist, bins = create_histogram(data, number_of_hist_bins)
    center = (bins[:-1] + bins[1:]) / 2.0
    plt.bar(center, hist, width = 0.03)
    plt.show()

# Applies thresholding for all values in dataset to reduce excess count for some values (0 steering angle mostly)
def apply_threshold_for_straight_driving(data, threshold_value = 500):
    
    # Calculate how many bins are present
    angles = np.array(data['steering'])
    number_of_unique_angles = np.unique(angles).shape[0]

    # Create histogram with that many bins
    hist, bins = create_histogram(data, number_of_bins = number_of_unique_angles)

    # Prepare array for indices for removal
    indices = []
    # For each bin
    for i in range(number_of_unique_angles):
        
        # Collect all indcies from current iteration
        temp_list = []

        # Find each steering angle that falls into current bin
        for j in range(len(data['steering'])):
            # If current angle falls into current bin
            if data['steering'][j] >= bins[i] and data['steering'][j] <= bins[i + 1]:
                temp_list.append(j)
        temp_list = shuffle(temp_list)
        # Cut steering angles up to threshold value
        temp_list = temp_list[threshold_value:]

        indices.extend(temp_list)

    # Now we got indices of all values that fall into bins that exceed threshold and should be removed
    data.drop(data.index[indices], inplace = True)
    return data

def get_image_paths_and_steering_angles(data, steering_correction_factor = 0.2):
    
    # Prepare lists for ret data
    image_paths = []
    angles = []

    for i in range(len(data)):
        # Lock dict on i-th item. This is equivavlent of single line from csv split
        fixed_item = data.iloc[i]

        # NOTE: Maybe flip a coin to check if 0.2 and -0.2 should be added
        center_im_path = fixed_item[0]
        left_im_path = fixed_item[1]
        right_im_path = fixed_item[2]
        steering = float(fixed_item[3])

        # Save path for center image
        image_paths.append(center_im_path)
        angles.append(steering)

        # Save path for left image
        image_paths.append(left_im_path)
        angles.append(steering + steering_correction_factor)
        
        # Save path for right image
        image_paths.append(right_im_path)
        angles.append(steering - steering_correction_factor)

    return image_paths, angles

def generator(image_paths, steering_angles, batch_size = 32, is_validation = False):
    # Total number of images in training set
    number_of_samples = len(image_paths)
    while True: # Inifinite loop

        # Prepare lists for images and steering angles
        X = []
        Y = []

        # For batch size
        for i in range(0, batch_size):

            # Choose random index
            index = randint(0, number_of_samples - 1)

            # Take steering angle and image at choosen index
            angle = steering_angles[index]
            image_path = image_paths[index]

            # Read image
            image = cv2.imread(image_path)

            # If generator is instantiated for training (not validation) then augment image
            if is_validation is False:
                image, angle = random_augment(image, angle)

            # Preproces image
            image = preprocess_image(image)

            # Add image and label into batch
            X.append(image)
            Y.append(angle)

        # Return batch of images and according angles
        yield np.array(X), np.array(Y)

################ Code starts here ################

# Load the dataset
data = load_data(data_dir)

# Display histogram of given dataset to check out balance
# display_histogram(data)

# Combat straight driving bias
thresholded_data = apply_threshold_for_straight_driving(data, 400)

# Checkout histogram after thresholding
# display_histogram(thresholded_data)

# Load image paths and appropriate steering angles
image_paths, steering_angles = get_image_paths_and_steering_angles(thresholded_data, CORRECTION_FACTOR)

# Split data to train and validation
train_im_paths, valid_im_paths, train_angles, valid_angles = train_test_split(image_paths, steering_angles, test_size = 0.2)

# NOTE: At this moment there are two really high peaks around -0.2 and 0.2, maybe this should also be tresholded
train_generator = generator(train_im_paths, train_angles, BATCH_SIZE)
valid_generator = generator(valid_im_paths, valid_angles, BATCH_SIZE, is_validation = True)

print('Total number of training samples: ', len(train_im_paths))
print('Total number of validation samples: ', len(valid_im_paths))

# Model definition
model = Sequential()
model.add(Conv2D(24, (5, 5), subsample = (2, 2), activation = 'elu', input_shape = (66, 200, 3)))
model.add(Conv2D(36, (5, 5), subsample = (2, 2), activation = 'elu'))
model.add(Conv2D(48, (5, 5), subsample = (2, 2), activation = 'elu'))
model.add(Conv2D(64, (3, 3), activation = 'elu'))
model.add(Conv2D(64, (3, 3), activation = 'elu'))
model.add(Flatten())
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(50))
model.add(ELU())
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(10))
model.add(ELU())
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(1))

# This is default learning rate
optimizer = Adam(lr=1e-3)

# Compile the model
model.compile(loss = 'mse', optimizer = optimizer)

# Print model summary
print(model.summary())

fitting_history = model.fit_generator(
                    train_generator,
                    steps_per_epoch = 300,
                    validation_data = valid_generator,
                    validation_steps = 200,
                    epochs= EPOCHS,
                    verbose = 1,
                    shuffle = True)

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