
# DeepSuperTracker and other implementations for MOT testing

This repository holds four different implementations of Multi-Object Trackers. Each one is an incremental improvement of the previous one, named as follows:

-  **Baseline IoU tracker**: Uses Faster-RCNN as object detector. Then those detections are compared against existing tracks in terms of Intersection over Union (IoU). The detection that maximizes the similarity for a certain track is assigned to that track.
- Hungarian algorithm tracker: An improvement over the baseline that uses the linear assignment method (Hungarian algorithm) to match detections with tracks.
- IoU + CNN features tracker: Uses a Deep Convolutional Neural Network (CNN) called EfficientNetV2, in the small version, to extract features for the image patches corresponding to the bounding boxes. Then a weighted sum between the cosine similarity and the IoU similarity is used to match detections to tracks.
- CNN features + IoU + Hungarian algorithm tracker: Uses CNN features (cosine similarity) + IoU for the cost matrix. Then, the Hungarian algorithm is applied to assign detections.

![ground_truth_vs_Detections](https://github.com/hector6298/deep_super_tracker/assets/41920808/8c7baf91-beb4-4661-a05b-73b1e1b725da)


## Getting started

The code in this folder is for the APPSIV practice. **The code here is just for grading purposes**. If you want to execute all the trackers, read the section "Running the code" first.

## Outline of the repository

The `tracker` folder, inside `src` is structured as follows:
- tracker: contains the code for our tracking classes.
    - abstract_tracker: Base class with boilerplate code for the custom trackers.
    - iou_tracker: Contains two classes of trackers based only on IoU distance. One using np.nanargmin and the other using linear assignment to assign detections to trackers.
    - deep_tracker: Contains two classes of trackers using IoU + CNN features. One using np.nanargmin and the other using linear assignment to assign detections to trackers.
    - track: Contains two classes to represent a track.
- detector: Code for object detection
- featurizer: Object to generate feature vectors for the embeddings. However, it was not used, but the code might be updated in the future to use this object.


## Running the code

For performance reasons, the notebook that executes the trackers is in Kaggle:

https://www.kaggle.com/code/hector6298/appsiv-p2-mot-tracking


To get a copy of the environment you have to do the following steps:

1. **Visit the Kaggle link**
2. Click on the **"Copy \& Edit"** button in the upper right corner of the web application. 
3. It already has setup steps prepared for testing.
4. The tester **must have a verified account** to have access to the data and execute using GPU accelerators. If you are not verified (via cellphone, for instance) you may have errors running the notebook.
5. Once the account is created and verified, click **Run All**. running the notebook using NVIDIA P100 GPU takes approximately 1 hour and 40 minutes.
