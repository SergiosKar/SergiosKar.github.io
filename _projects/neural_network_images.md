---
layout: project
title: Master Thesis
featured-img: mnist
project-url: https://github.com/SergiosKar/Convolutional-Neural-Network
category: Deep Learning
---

As part of my thesis during my MEng degree in Electrical and Computer Engineering , we developed a Computer Vision library that 
allows the user to recognize objects in images using deep learning. To accomplish that, the user/developer can 
define his own neural network architecture and train his own images on it. 

The system supports **fully connected and convolutional neural networks** , which we implement in C++ from scratch. 
To speed up the training, we decided use parallelization and execute the training in GPU, which we programmed 
with the OpenCL library. Also OpenCV was used to parse and read the images and do all the necessary preprocessing of the dataset.



