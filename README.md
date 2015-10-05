**DISCLAMER:**

This project was created within an academic research setting, and thus should
be considered as EXPERIMENTAL code. There may be bugs and deficiencies in the
code, so please adjust expectations accordingly. With that said, we are
intrinsically motivated to ensure its correctness (and often its performance).
Please use the corresponding web repository tool (e.g. github, bitbucket, etc)
to file bugs, suggestions, pull requests; we will do our best to address them
in a timely manner.


**LAYOUT:**
- face_detection/
  - cfg/:                 dynamic_reconfigure configuration files
  - include/:             cascade files for the detection
  - launch/:              roslaunch files
  - src/:                 source files
  - CMakeLists.txt:       CMake project configuration file
  - LICENSES:             license agreement
  - package.xml:          ROS/Catkin package file
  - README.txt:            this file


**Why use this package:**
This ROS node is designed to detect faces in images coming from a ROS image
topic. Currently this node displays or publishes an image with the resulting
detections drawn on top of the image. The settings of the detection system
can be easily adapted using ROS rqt_reconfigure.

**How to Use this package:**
To run the package, use the provided RosLaunch file as follows:
  - roslaunch face_detection face_detection.launch

The settings of the program can be changed with the ROS rqt_reconfigure setup.
  - rosrun rqt_reconfigure rqt_reconfigure

Once you have the rqt_reconfigure open, change the input image default
(/camera/image_raw) to your desired input. Additional information about the
different settings are annotated in the dynamic reconfigure setup (hover over
the setting in the rqt_reconfigure for additional information)

**Future plans:**
This package is currently still under development and will shortly support
the publication of the face coordinate.



Copyright (c) 2015, Philippe Ludivig

All rights reserved.

BSD2 license: see LICENSE file
