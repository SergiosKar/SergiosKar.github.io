---
layout: post
title: Neural Network from scratch-part 1
summary: How to buld a neural network library using C++ and OpenCL
featured-img: nn
---

# Neural network library from scratch(part 1)

## Fully Connected Neural Network

Let's build a neural network from scratch. I mean why not?
You may say : Pff... Big deal.. With Python and Numpy is just a matter on hours. What if I told you that i will use C++. Nah I'm kidding. I am going to use C.

The reason for that is that i want to train my network on GPU and GPUs dont understand Python, not even C++. My plan is to use OpenCL along with C++ to build a fully functional library to create your own Neural Network and train it. And to spice it up a little , why not implementing a convolutional neural netwok instead of a simple, boring Fully Connected NN. But first things first.

Let's not dive immediately GPU's kernel code. First we should build our library skeleton.

```C++
// First initialize OpenCL
OpenCL::initialize_OpenCL();

//Create vectors for input and targes
std::vector<std::vector<float> > inputs, targets;
std::vector<std::vector<float> > testinputs;
std::vector<float> testtargets;

//Define our neural network
ConvNN m_nn;
std::vector<int> netVec; 
netVec = { 1024,10 };
m_nn.createFullyConnectedNN(netVec, 1, 32);

//Train the network
 m_nn.trainFCNN(inputs, targets, testinputs, testtargets, 50000);

//Test accuracy on test data
m_nn.trainingAccuracy(testinputs, testtargets, 2000, 1);

```

Ok thats the ordinary process of every machine learning pipeline with the difference that instead of Sklearn or Tensorflow functions, here we have C++. Quite the accomplishment! Right? 

So far so good. We have the baseline of our software. Now it is time to develop the actual structure of a neural network. The basic entity of any NN is the Node and many nodes stacked together form a layer. There you have it:

```C++
typedef struct Node {

    int numberOfWeights;
	float weights[1200];
	float output;
	float delta;

}Node;

typedef struct Layer {

	int numOfNodes;
	Node nodes[1200];

}Layer;
```

Since this is plain C, we cant use an std::vector and we need plain C because the abode will be compiled and executed by the actual GPU. But we're getting there.  Please noe that a better way than an array with predefined length would be to malloc the necessary space in memory every time, but that is for some other time.  
