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
---
## Installation

Prerequisites
- Python 3.4+
- (Optional) venv

1. Clone this repository using  
```git clone https://github.com/YosYohannes/pupil_centroid_opencv```  
1. Go to the git directory  
```cd pupil_centroid_opencv```  
1. Create virtual environment  
```virtualenv venv```  
1. Activate virtual environment  
MacOS / Linux  
```source venv/bin/activate```  
Windows  
```venv\Scripts\activate```  
1. Install packages  
```pip install -r requirements.txt```  

### Getting Started

Run algorithm with sample video:  
```python pupil_detect.py -p test_videos\sample.mkv```  
When installed correctly, the video will play with pupil centroid marked

![alt text](https://github.com/YosYohannes/pupil_centroid_opencv/blob/main/assets/frame.PNG)

To run with heavier smoothing but less responsive:  
```python pupil_detect.py -p test_videos\sample.mkv -s```

![alt text](https://github.com/YosYohannes/pupil_centroid_opencv/blob/main/assets/double.PNG)

To display layer view in a double frame:  
```python pupil_detect.py -p test_videos\sample.mkv -d```


While video is playing use `spacebar` to play/pause the video.
While paused, use `k` to go to next frame.
Use `esc` to end video.

At the end, script will print a list of centroids position with the same lenght as number of video frames analysed, `(-1, -1)` is value returned for no pupil centroid found

### Usage

To use, add model and function file to your project directory
> haarcascae_eye.xml  
> yos_pupil_detection.py

In the code, import the function and call it to get list of centroid position
```
from src.yos_pupil_detection import get_centroid

video_file = "some file path"
centroids = get_centroid(video_file)
```

For more info use
`help(get_centroid)`


## Methods

### Step 0: Resize
First to improve consistency, scale the frame to have at least 1280 px wide and 960 px high. This works under the assumption that video's aspect ratio is not extreme (e.g. 20:1). Odd aspect ratio is untested and may results in slow processing

### Step 1: Find the eye
To remove false detection of pupil, the algorithm first detect the eye by greyscaling and using a cascade classifier found in [this](https://github.com/HassanRehman11/Pupil-Detection) pupil detection algorithm. We then cut the bounding box by 15% in all direction to reduce the total number of pixels to run further processes on.

To overcome issues with some failure to detect eye, a kalman filter is deployed, keeping track of eye's center, speed and size. This will have an expiration of 0.5 seconds.

### Step 2: Find pupil
Start with a common step of blurring the image. Initially, adaptive thresholding and dilation was explored to enhance the image for pupil fiding, but these are dropped. I found increasing the contrast of the image works sufficiently well so we can keep this step simpler.

To find the pupil I use Hough Circle method. To reduce the processing time, the circle size are bound to be between 80-200 px for the intial search. Depending onn the usecase and camera setup this may need to be changed. To reduce jitter and further cut processing time, I bound the circle radius further to +- 6% from previously found circle. This tighter radius restriction has an expiration of 0.5 second as well.


### Step 3: Smoothing
In "smooth" mode, smoothing is achieved by taking a 0.4 s  average of the pupil position. This will result in a smooth but less responsive centroid detection. In "snap"/default mode, smoothing is achieved by taking a 0.8 s frame average only when the position is relatively stationary, otherwise, no averaging is done to ensure responsiveness.

What about using another Kalman Filter for this?
I did experiment a bit on using it, but did not think the behaviour is suitable. I think it is all based on use case. I try to achieve as little false positive as possible, so not using a prediction may serve that purpose better. As for smoothing, the eye (or rather the pupil) usually turn to face down when blinking. With less frame resolution, the filter will usually detect a very high downwards velocity and ended up with a rather far prediction. With tuning and safeguards, this method is probably preferrable for its generality.

### Step 4: Drawing and Output
Output handling, adding drawings to both showable views

## Performance

In the sample test, the video output ran with a `19 fps` average for single view.
With both views being displayed, it ran with `18.4 fps` average

On smooth mode, there are very little jitter, but it takes around 0.5 seconds for centroids to be in the highly accurate position when it is moving around fast

On default mode, there are slightly more jitter especially during movement. But is able to maintain a more accurate centroid position.

Detection rate works best when pupil size is around 80-150 px in radius. Zooming and out can be managed by increasing the search range at cost of more processing.

<pre>
  /)/)  
 (o.o)  
o(")(") 
</pre>
 