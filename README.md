# **Behavioral Cloning** 

![Simulator][image8]

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./images/arch.png "Model Visualization"
[image2]: ./images/sharp-left.jpg "Sharp left"
[image3]: ./images/sharp-right.jpg "Sharp right"
[image4]: ./images/recovery.gif "Recovery"
[image5]: ./images/training_loss_06-05-2020-19-46-58.png "Loss final"
[image8]: ./images/ss-small.png "Cover"
[image9]: ./images/pre-balancing.png "histo balanced"
[image10]: ./images/after-balancing.png "histo treshold"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network (Keras 2.1.2 and TF 1.8.0)
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 showing video of car driving one lap fully autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

## General Model Architecture and Training Strategy

Model consists of a convolution neural network with 3x3 and 5x5 convolutional layers, where depths of these layers range from 24 to 64.

The model includes both RELU and ELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Additional data is cropped to isolate region of interest

Model does not include dropout layers, but instead uses small number of epochs in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting - 20% of dataset is reserved for validation, and 80% was training set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually it was left at default value of 0.001.

For training data, initially, dataset provided in lessons was used. Model showed solid results on that dataset, but there were some problematic sections of the track where car would run out of drivable section of track. In order to tackle this problem additional dataset was created. New dataset consisted mostly of recovery maneuvers. Combination of these two sets also did not prove to be enough balanced new third and final dataset was collected which consisted of three laps in each direction and one recovery lap with different recovery maneuvers also in both directions.

For details about how I created the training data, see the next section. 

---

### Detailed Model Architecture and Training Strategy

The overall strategy for deriving a model architecture was to use well known and field proven models with possible minor modifications.

LeNet architecture was used as a strating point. 
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high MSE on both training set and validation set. This implied that the model was underfitting. 

Next step included moving to more powerful and deeper model. Suggested [NVIDIA End2End model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was used here and it proved to be much more efficent that previous model.

This model proved to have much lower MSE then previous one, and also had better driving performance, since model could complete track at this point with couple of interventions to prevent running on out track.

Here is experimented with different values of hyper parameters such as batch size, number of epochs, dropout rate, different activation functions and also data augmentation.

At this point, hyperparameter tweaking was not enough to make car drive whole track on it's own, and model started overfitting. Data augmentation helped a bit here, since it doubled data set. Each image in dataset was flipped w.r.t. y axis. In addition to this, steering wheel angle was also 'fliped' by multiplying it's value by -1. This improved results, but few car still had problems staying on track in sharp turns and areas where road boundary was not clear (dirt sections).

New dataset was created in order to tackle these problems. Main focus of dataset was to introduce recovery maneuvers, but also to remove left turn bias track had.

In ordert to tackle left turn bias additional dataset was collected. This dataset consisted of three laps in each direction and one lap with recovery maneuvers in each direction. This dataset proved to be much more balanced.
On following image histogram of present steering angles is shown.

![alt text][image9]

On this histogram we can see that data is no longer left turn biased, but new problem emerged, and that is straight driving bias. This problem was tackled by applying thresholding technique as part of dataset preprocessing (`model.py` function `apply_threshold_for_straight_driving()`). After additional balancing histogram looked much better.

![alt text][image10]

In addition to having balanced dataset, additional augmentaiton was applied which included random slight zooming (up to 30%), slight moving (translation) and brightnes reduction/increase, beside standard flipping.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

As already mentioned, [NVIDIA End2End model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was choosen as final model for this project. Model consists of 5 convolutional layers and 3 fully connected layers, with single output, and predicts steering wheel angle based on input acquired from single camera frontal camera. Model's architecture can be seen on image bellow

![alt text][model]

Model is created in model.py file and code can be seen in following snippet.

```python
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
```

Model sumarry:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300    
_________________________________________________________________
elu_1 (ELU)                  (None, 100)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
elu_2 (ELU)                  (None, 50)                0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
elu_3 (ELU)                  (None, 10)                0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________

Input to model is first normalized in order to shift data to have small mean close to zero, but also image is converted to YUV color space as in original paper, ROI is cropped and input image is resized to 200x66x3 in order to fit input size from original paper. This is not done in model itself, but instead in `model.py` in separate function `preprocess_image()`. This function is also applied on deployment of model in `drive.py` before input is fed into model.

After that image is passed through convolutional layers. Stride of 2 was choosen for first 3 convolutional layers, which effectivly reduces size of input further. Last two convolutional layers do not further reduce size of features. In convolutional layers ELU activation was used in order to combat dying neurons problem. After this, output is flatten, and fully connected part of network starts here. In this section of network, ELU activation was choosen since it provided better result and less 'shakey' controlls.

Dropout layeres were initially introduced in fully connected section of network, but they proved not be really helpful since they model had less smooth controls when dropout layers were used in trainig. In addition to this higher number of epochs had to be used with them and results were almost the same, but training time was much higher and loss was droping way more slowely even for low dropout rates. After new dataset was introduced and after it was preprocessed, it's size dropped significantly 10536 samples. Now dropout layers had to be used, even data augmentation techniques were introduced in order to prevent overfitting. Dropout rate was set to 30%.

Generator was used during training phase of this model. It was mainly used in order to save RAM, but also to do random data augmentation on the fly. When called, generator would return batch of images and appropriate labels where size is defined by batch size. Generator is implemented in `model.py` in `generator()` function. Same generator was used for trainign and validation samples feeding, where for validation samples, augmentation was not applied on the images. Images and steering wheel angles are chosen by uniformly and randomly from dataset. After one sample is selected, it is randomly augmented and packed inside of batch which is then fed to model.

#### Creation of the Training Set & Training Process

During training of model i have initially used only dataset provided in leassons prior to project, but ended up creating my own dataset. Since driving on the middle of the track had preety good with usage of just lesson dataset, I have focused on car recoveries and sharp turns while i was collecting additional data. Biggest part of my dataset consisted of recovery maneuvers where car goes from side of the road back to the middle. In addition to this i have recorded multiple scenraios where i saw car running of the track. This includes sharp turnes like one after bridge, where additional difficulty is lack of clear boundary of the track, since this is the place where track is connected with dirt road.

![alt text][image2]
![alt text][image3]

In scenraios like this, steering wheel angle value had to be high in order to avoid running of the track, and this proved to be very useful asset to the dataset, since first dataset was lacking these situations and car was not able to recover from these scenarios.

Short demonstration of these scenraios can be seen on gif bellow. Recording starts when vehicle is near to the border of the road, and than sharp turn is made in order to recover back to the middle of the road.

![alt text][image4]

Final dataset consisted of three laps of center drving in both directions and one lap of recovery manuevers previously mentioned, also in both directions. This proved to be very balanced dataset and when it was combined with mentioned data augmentation techniques  proved to be sufficent with around 10.5k samples.

As already mentioned around 20% of dataset was used for validation and rest for training process. I have also experimented with different values for batch sizes, and found that 100 batch size worked best in my case. Choosen steps per epochs is 300 training steps. This means that 3000 images are fed in single epoch to the model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 because dropout was used with around 30% dropout rate, so overfitting was avoided. Loss over these epochs can be seen on image bellow:

![alt text][image5]

When dropout layers were included in model, loss was reducing slower, so more epochs had to be introduced.

Adam optimizer was used, so learning rate was not set manually and it was left at default 0.001 value.

Video of car driving one cycle autonomously can be seen [here](https://www.youtube.com/watch?v=dDdehBeF_eI&feature=youtu.be).

### Drawbacks and possible improvements.

Model still does not have fully smooth controls, and car can sometimes wander of from track center. Model is not able to finish second track as uphill road and very sharp turns seem to be problematic at this stage. Collecting bigger and more balanced dataset and also some parameter tweaking could potentially bring model to completition of second track.
