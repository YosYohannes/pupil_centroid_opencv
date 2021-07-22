# OpenCV Find Pupil Centroid

This is a solution submission to [test_name: FindPupilCentroid](https://github.com/lackdaz/cv_find_pupil_centroid)
Authored by [Seth Loh](https://github.com/lackdaz) (c) 2020

## Objective

The goal is to **develop a pupil detection algorithm to detect the centroid of the pupil** that is lightweight and reliable.

### Key Objectives  

1. Measure within 10 fps between frame computations.
1. Detect the centroid without jitter 
1. Reject false centroids
1. Maintain a detection rate of 80%

## Instruction

### Installation

### Running example

### Using the function

## Methods

## Step 0: Resize
First to improve consistency, scale the frame to have at least 1280 px wide and 960 px high. This works under the assumption that video's aspect ratio is not extreme (e.g. 20:1). Odd aspect ratio is untested and may results in slow processing

## Step 1: Find the eye
To remove false detection of pupil, we first detect the eye by greyscaling and using a cascade classifier found in [this](https://github.com/HassanRehman11/Pupil-Detection) pupil detection algorithm. We then cut the bounding box by 15% in all direction to reduce the total number of pixels to run further processes on.

## Step 1.5: Prepare image for pupil finding
Start with a common step of blurrign the image. Initially, adaptive thresholding and dilation was explored to enhance the image for pupil fiding, but these are dropped. I found increasing the contrast of the image works sufficiently well so we can keep this step simpler

## Step 2: Find pupil
To find the pupil I use Hough Circle method. To reduce the processing time, the circle size are bound to be between 80-200 px. Depending onn the usecase and camera setup this may need to be changed. To reduce jitter, I bound the circle radius further to +- 5 px from previous frame circle if found. This additional bound will not be necessary with the implementation of Kalman Filter


## Step 3: Smoothing
In "smooth" mode, smoothing is achieved by taking a 10 frame average of the pupil poition. This will result in a smooth but less responsive centroid detection

In "snap" mode, smoothing is achieved by taking a 20 frame average only when the position is relatively stationary, otherwise, no averaging is done to ensure responsiveness.

This will be replaced with implementation of Kalman Filter.

## Step 4: Drawing
Output handling

## Step 5: Handling of disjoint data association
Area of the eye will be kept for 0.5 seconds or 12 frames. Within this limit, pupils will still be searched in last known area. This is not the most accurate and will be replaced with Kalman Filter implementation

### Asssesment Critera

1. Completion of key objectives [60%]
1. Code quality [20%]
1. Packaging [20%]
