# Single Person Tracker

## Description

This script employs a variety of computer vision and deep learning techinques to track a person throught a video.

It is designed to use multiple crops of a person in order to extract relevant features.

As opposed to other trackers it does not use a Kalman Filter to estimate the trajectory and velocity of a person. It is purely based on extracting visual features and searching for them throughout the video frames.

## Script Description

* `demo_script_video_parallel.py`
  * main script
  * reads refference images from `ref_images`
  * outputs result video into `output`
  * extract features from reference images and from each detection
  * compares detection features with reference features
* `feature_extraction.py`
  * contains the FeatureExtractor object
    * human segmentation
    * color histogram calculations
    * feature extraction via neural net models
    * human limb separation
* `feature_comparison.py`
  * contains the Comparator object
    * cv2 histogram comparison
    * cosine similarity (torch and sklearn)
    * color clothing comparison (not used)
    * ORB feature matching (not used)
* `human_separation.py`
  * contains the HumanSeparator object
    * handles the detection extraction from video
    * handles the video capture management
    * handles detection file management
* `utils.py`
  * utility functions
    * loading segmentation model
    * coordinate transformations (xyxy -> xywh)
    * converting RGB values to color names (not used)
    * applying brigthness and contrast to frame (not used)
    * drawing custom designed detection borders
    * calculating optimal font scale based on detection size
    * converting frame number to time and time to frame number
