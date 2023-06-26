
# DeepSuperTracker and other implementations for MOT testing

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

The code for the tracker is hosted on Google Drive so that it can be installed remotelly and seamlessly. It can be accessed through the following link (if you want to visit it. You dont have to do anything with it. The process does the setup for you):

https://drive.google.com/drive/folders/1R0xgOiVEpf8_EwviQ72NZ2_9b_2wCcYj?usp=share_link


Furthermore, due to performance reasons, the notebook that executes the trackers is in Kaggle:

https://www.kaggle.com/code/hector6298/appsiv-p2-mot-tracking


To get a copy of the environment you just have to **visit the Kaggle link**, then click on the **"Copy \& Edit"** button in the upper right corner of the web application. 

It already has setup steps prepared for testing. After the notebook is copied in you own workspace, you can just hit **"Run All"**. Moreover the tester **must have a verified account** to have access to the data and execute using GPU accelerators. If you are not verified (via cellphone, for instance) you may have errors running the notebook. Once the account is created and verified, running the notebook using NVIDIA P100 GPU takes approximately 1 hour and 40 minutes.
