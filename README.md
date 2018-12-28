## Intro

[![Build Status](https://travis-ci.org/thtrieu/darkflow.svg?branch=master)](https://travis-ci.org/thtrieu/darkflow) [![codecov](https://codecov.io/gh/thtrieu/darkflow/branch/master/graph/badge.svg)](https://codecov.io/gh/thtrieu/darkflow)

Real-time object detection and classification. Paper: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).

Read more about YOLO (in darknet) and download weight files [here](http://pjreddie.com/darknet/yolo/). In case the weight file cannot be found, I uploaded some of mine [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU), which include `yolo-full` and `yolo-tiny` of v1.0, `tiny-yolo-v1.1` of v1.1 and `yolo`, `tiny-yolo-voc` of v2.




## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3. PYkinect2

 bin/tiny-yolo.weights --json
 
 This uses Pykinect2 to get the color and depth streams from the Kinect V2.
 Then the color stream is feed to the YOLO model.
 This uses TINY-YOLO beacaus my machines is a slug:
 Then it takes the X,Y cordinates of the centers of  the bounding boxes and
 Uses those cordinates to find the depth data from the depth stream.
 Then it adds the depth data to bounding box label.
 It also puts a colored circle(the same color as the bounding box)
 on the depth data window where it is readinf the depth data pixel.


```


