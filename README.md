# Orthosis-Control
A Convolutional Neural Network powered control system for a hand orthosis/exoskeleton. 

This repository holds files related to a graduation assignment for the Saxion University of applied sciences.
The MSWC dataset is used in combination with the MobileNetV3 architecture for the CNN.


# Folder structure
The folders have the following purposes:

**Dataset**:                     Holds the dataset that is used. The dataset consits of certain parts/words of the MSWC dataset.

**On-device inference**:         Holds all the needed files for the inference that is used on the Raspberry Pi 4B. A trained model is also needed but not included

**Pre-processing&Training**:     Holds pre-processing scripts for the MSWC dataset and the notebook that is used for training the model.

**Trained models**:              Holds several pre-trained models.

# Setup guide
This guide will describe the process of running a pre-trained model onto a new Raspberry Pi 4B.
The following things are needed:
-Raspberry Pi 4B + USB-C power supply
-Minimum 16GB of storage, so either a flashdrive or SD card(Flashdrive preffered due to read-write cycles damaging SD card)
-Mini displayport -> HDMI/Displayport cable
-IQaudio Codec Zero ADC HAT
-Secondary monitor + Keyboard + Mouse for RPI

**Flashing image to Raspberry Pi**
We start by flashing an image which holds an OS to the Raspberry Pi. We will use an existing image which includes the TensorFlow library, which will make instalation easier.
Download the QEngineering 64-bit Raspberry Pi OS image: https://drive.google.com/file/d/1s8ulI44O96qmVPmWyz8yw3lzamh-32gN/view?usp=sharing
Download and install the image flasher: https://downloads.raspberrypi.org/imager/imager_latest.exe

Insert the storage device into your computer
Open the Raspberry Pi imager
Go to: Select OS, and scroll down and select the custom image file.
![image](https://user-images.githubusercontent.com/42100039/160374489-421ad84e-0802-4c09-8e80-84c354f69840.png)
![image](https://user-images.githubusercontent.com/42100039/160374566-718dac19-fa1b-47c0-a4e5-2a126f687d52.png)

Select the recently downloaded QEngineering Raspberry Pi OS image, and press Open.

Lastly select the right storage device(which should be the inserted SD or flashdrive), and press the write button in the imager. 

Let the process finish, which will result in a flashed storage device.


