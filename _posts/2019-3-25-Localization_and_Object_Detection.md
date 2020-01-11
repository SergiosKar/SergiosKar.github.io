---
layout: post
title:   Localization and Object Detection with Deep Learning 
summary: Explain RCNN, Fast RCNN and Faster RCNN
featured-img: regions_proposals
redirect_to:
  - https://theaisummer.com/Localization_and_Object_Detection
---

# Localization and Object Detection with Deep Learning (part 1)

Localization and Object detection are two of the core tasks in Computer Vision ,
as they are applied in many real-world applications such as Autonomous vehicles
and Robotics. So, if you want to work in these industries as a Computer vision
specialist or you want to build a relative product , you better have a good
grasp of them. But what are they? What Object detection and localization means?
And why we group them as they are one thing?

First things first. Let’s do a quick recap of the most used terms and their
meaning to avoid misconceptions:

-   **Classification/Recognition**: Given an image with an object , find out
    what that object is. In other words, classify it in a class from a set of
    predefined categories.

-   **Localization** : Find where the object is and draw a bounding box around
    it

-   **Object detection**: Classify and detect all objects in the image. Assign a
    class to each object and draw a bounding box around it.

-   **Semantic Segmentation**: Classify every pixel in the image to a class
    according to its context, so that each pixel is assigned to an object

-   **Instance Segmentation**: Classify every pixel in the image to a class so
    that each pixel is assigned to a different instance of an object

Remember, though, that these terms are not clearly defined in the scientific
community, so you may encounter one of them in a different meaning. In my
understanding, these are the correct interpretations.

As we get the basic terms straight, it is time to do some localization and
object detection. How do we do it? Well there have been many approaches over the
years, but since the arriving of Deep Learning, Convolutional Neural Networks
became the industry standard. Remember **our goal is to classify the object and
localize it**. But are we sure that there is only one object? Is it possible
that there are two or three or fifteen objects? In fact, most of the time it is.

That’s why we can split our problem into two different problems. In the first
case , we know the number of objects (we will refer to the problem as
classification + localization) and in the second we don’t (object detection). I
will start with the first one as it is the most straightforward.

![cv_tasks]({{"/assets/img/posts/cv_tasks.jpg" | absolute_url}})


>>> [Stanford University School of Engineering](https://www.youtube.com/channel/UCdKG2JnvPu6mY1NDXYFfN0g)

## Classification + Localization

If we have only one object or we know the number of objects, it is actually
trivial. We can use one convolutional neural network and train it **not only to
classify the image but also to output 4 coordinates for the bounding box**. **In
that way we treat the localization as a simple regression problem**.

For example, we can borrow a well-studied model such as ResNet or Alexnet which
consists of a bunch of convolutional, pooling and other layers, and repurpose
the fully connected layer to produce the bounding box apart from the category.
It is so simple that make us question whether or not it will give results. And
it actually works pretty well in practice. Of course, you can get fancy with it
and modify the architecture for serving specific problems or enhance its
accuracy, but the main idea remains.

Be sure to note that in order to use this model, we should have a training set
with images annotated for the class and the bounding box. And it is not the most
fun to do such annotations.

But what if we do no know the number of objects a priori? Then we need to get
into the rabbit’s hole and talk about some hardcore stuff. Are you ready? Do you
want to take a break before? Sure, I understand but I warn you not to leave.
This is where the fun begins.

## Object Detection

I am kidding. There is nothing hardcore about the architectures which will be
discuss. All there is, are some clever ideas to make the system intolerant to
the number of outputs and to reduce its computation cost. So, we do not know the
exact number of objects in our image and we want to classify all of them and
draw a bounding ox around them. That means that the number of coordinates that
the model should output is not constant. If the image has 2 objects , we need 8
coordinates . If it has 4 objects, we want 16. So how we build such a model?

One key idea to traditional computer vision is regions proposal. We generate a
set of windows that are likely to contain an object using classic CV algorithms,
like edge and shape detection and we apply only these windows( or regions of
interests) to the CNN. To learn more about how regions are proposed, make sure to check 
[here](https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/).
 
This is the basis on a fundamental
[paper](https://arxiv.org/abs/1311.2524) , which introduced a new architecture
called RCNN.

![regions_proposals]({{"/assets/img/posts/regions_proposals.jpg" | absolute_url}})


### R-CNN

Given an image with multiple objects , we generate some regions of interests
using a proposal method( in RCNN’s case this method is called selective search)
and warp the regions into a fixed size. We forward each region to Convolutional
Neural Network (such as AlexNet), which will use an SVM to make a classification
decision for each one and predicts a regression for each bounding box. This
prediction comes as a correction of the region proposed, which may be in the
right position but not at the exact size and orientation.

![rcnn]({{"/assets/img/posts/rcnn.jpg" | absolute_url}})


Although the model produces good results, it suffers from a main issue. It is
quite slow and computational expensive. Imagine that in an average case, we
produce 2000 regions, which we need to store in disk, and we forward each one of
them into the CNN for multiple passes until it is trained. To fix some of these
problems, an improvement of the model comes in play called ‘fast-RCNN’

### Fast RCNN

The idea is straightforward. Instead of passing all regions into the
convolutional layer one by one, we pass the entire image once and produce a
feature map. Then we take the region proposals as before ( using some external
method) and sort of project them onto the feature map. Now we have the regions
in the feature map instead of the original image and we can forward them in some
fully connected layers to output the classification decision and the bounding
box correction.

![fastrcnn]({{"/assets/img/posts/fastrcnn.jpg" | absolute_url}})

Note that the projection of regions proposal is implemented using a special
layer( ROI layer) ,which is essentially a type of max-pooling with a pool size
dependent on the input, so that the output always has the same size. For more
details on the ROI layer check this great [article](https://deepsense.ai/region-of-interest-pooling-explained/).

### Faster RCNN

And we can take this a step further. Using the produced feature maps from the
convolutional layer, we infer regions proposal using a Region Proposal network
rather than relying on an external system. Once we have those proposal , the
remaining procedure is the same as Fast-RCNN (forward to ROI layer, classify
using SVM and predict the bounding box). The trick part is how to train the
whole model as we have multiple tasks that need to be addressed:

1.  The region proposal network should decide for each region if it contains an
    object or not
2.  And it needs to produce the bounding box coordinates
3.  The entire model should classify the objects to categories
4.  And again predict the bounding box offsets

If you want to learn more about the training part you should check the original
[paper](https://arxiv.org/abs/1506.01497), but to give you an overview we need
to utilize a multitask loss to include all 4 tasks and back propagate this loss
to the network.

![fasterrcnn]({{"/assets/img/posts/fasterrcnn.jpg" | absolute_url}})


As the name suggests, FasterRCNN turns out to be much faster than the previous
models and is the one preferred in most real-world applications.

Localization and object detection is a super active and interesting area of
research due to the high emergency of real world applications that require
excellent performance in computer vision tasks (self-driving cars , robotics).
Companies and universities come up with new ideas on how to improve the accuracy
on regular basis.

There is another class of models for localization and object detection, called
single shot detectors, which have become very popular in the last years because
they are even faster and require less computational cost in general. Sure, they
are less accurate, but they are ideal for embedded systems and similar
power-hungry applications.

But to learn more , you have to wait for my next article…
