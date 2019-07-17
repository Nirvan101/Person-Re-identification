# Person-Re-identification
## Introduction

This is a project for detecting and tracking people in a video and making a list of all unique people that have appeared in the video. If a person appears for some frames and then leaves the video for some time and then reappears later, the model should recognize it's the same person and not count this person for a second time.
I have combined 2 approaches: Tracking and Re-identification.

![](https://cdn-images-1.medium.com/max/1200/1*-WkySYuR7koWY3g_Ikec2A.gif)

[Additional documentation](https://docs.google.com/document/d/1JsVFL44qAQoDbLO2NOSjiuXVBEDEELg-9uv4PNACDLc/edit#)

## Output
This model is run on a video which contains people in it. If a total of N people are seen in the video, the output is a list of N folders, each belonging to one person. Each folder contains cropped images of that person from all frames in the video.
In the following screenshot, there are 7 folders as 7 people have been detected in the video.

![7 folders created for 7 detected people](https://user-images.githubusercontent.com/23417993/51552197-10f61080-1e96-11e9-9966-d7a75e801dd1.png)
![Folder for first person](https://user-images.githubusercontent.com/23417993/51552198-10f61080-1e96-11e9-8bb3-73c8a11d64e6.png)


## Setup

This code uses the reidentification model from [this paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf) and the code can be obtained from:

```
git clone https://github.com/thomaspark-pkj/person-reid.git
```

This model first needs to be trained for image reidentification. Download CUHK03 dataset from [CUHK Person Re-identification Datasets](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) then extract file.
```
python cuhk03_dataset.py your_dataset_path
```

To train:
```
python run.py --data_dir=your_dataset_path
```

To test it: 
```
python run.py --mode=test --image1=your_dataset_path/labeled/val/0000_00.jpg --image2=your_dataset_path/labeled/val/0000_05.jpg
```

If the images are of the same person, it will return ```True``` otherwise ```False```.

Now replace the ```run.py``` file with OUR ```run.py``` file. This new file is slightly changed to work together with the rest of our code.

Now place the ```human_detect_track2.py``` file and the ```model``` folder in the current folder alongside ```run.py```. Place a video file called ```video2.avi``` also in this folder. 

Make an empty folder called ```past_ppl``` also in the current folder.

```
python human_detect_track2.py video2.avi   
```

NOTE: The folder ```past_ppl``` must be created and empty before running the code each time.

This runs the complete model on the given video. The people in the video are detected and tracked. Bounding boxes appear around all the detected people in the video. The model tracks people throughout the video and re-identifies people when they disappear and reappear multiple times, thereby not counting them multiple times. The folder ```past_ppl``` is filled with folders of detected people. If n people are detected and tracked in the video, n folders and created- each having all the images of that person from the video. For instance, when the ith people is detected, he will have a folder called `i` created and images of him will be cropped out of all the frames he appeared in, and they'll be stored in the folder i. 

So, this model will make a list of unique people in the video and will store their individual video clips in separate folders.


## Models and approaches


### Tracking


Between 2 consecutive frames, the person would move very little and his bounding boxes from those 2 frames will overlap a lot. I used this property to track people. People once detected are simply tracked across the video by checking the degree of overlap with the bounding boxes of the previous frame. If the bounding box of a person in a frame overlaps a lot with a box from the previous frame- we can conclude they are the same person. People are tracked this way.

```
python human_detect.py
```

To improve tracking, we can use the last k(=25) frames instead of just the previous frame. The current bounding box is checked for overlapping with all boxes in the previous k frames. Hence, tracking is done over k frames.

```
python human_detect_track.py
```

### Tracking + re-identification


This model uses both tracking and re-identification and is an extention of ```human_detect_track.py```. The people are tracked using intersection-over-union between frames. However, in the case that a person enters the video and was not present in the previous frame or the tracking failed for some reason, then the re-identification model in ```run.py``` is run to determine who this person is and whether he is a new person or was seen before.

The re-identification neural net is located in ```run.py```. It receives images of 2 people and returns whether they are the same person or not. So, when a person in the video is not tracked- his identity is found by using reidentification to compare him with all the other people.

In this version, the model iterates over all frames and in each frame it obtains the person bounding boxes. But it doesn’t re-identify each person, only the uncertain ones. The model stores a list of bounding box positions from the previous k frames. It matches the bounding boxes from the current frame to the previous frame boxes that are very close in position. In other words, if the current bounding box is very close to a bounding box from a previous k frames- that means that it’s the same person who has just slightly moved between frames. This way the model identifies each person only once (when they’re first seen) using the neural network. For the next frames, it just tracks the person.

So, for each bounding box from the current frame, the model tries to find a bounding box from the previous frame which greatly overlaps with it (IOU > 0.9). If such a box is found, the model assigns the previous box’s person to the new bounding box. In this way, people are identified without actually running the neural network on them. If a bounding box is not able to match with any previous box, it means that this person just entered the frame and was not there in the previous frame. In this case, the reidentification neural network is run on the person to determine if he has appeared before or if he is a new person totally, in which case he is added to the list of unique people detected. This new person is compared with all the other previously detected people. If a match is found, then that means the person had appeared before but then disappeared for an intermediate period. If a match is not found, that means the person is appearing in the video for the first time and needs to be added to the list of unique people.

```
python human_detect_track2.py
```
