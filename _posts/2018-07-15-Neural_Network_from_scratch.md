---
layout: post
title: Neural Network from scratch-part 1
summary: How to buld a neural network library using C++ and OpenCL
featured-img: NN
---

# Neural network library from scratch(part 1)

## Fully Connected Neural Network

Let's build a neural network from scratch. I mean why not?
You may say : Pff... Big deal.. With Python and Numpy it's just a matter of hours. What if I told you that i will use C++. Nah I'm kidding. I am going to use C.

The reason for that is that i want to train my network on GPU and GPUs don't understand Python, not even C++. My plan is to use OpenCL along with C++ to build a fully functional library to create your own Neural Network and train it. And to spice it up a little , why not implementing a convolutional neural netwok instead of a simple, boring Fully Connected NN. But first things first.

Let's not dive immediately on GPU's kernel code. First we should build our library skeleton.

```c
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

```c
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

Since this is plain C, we can't use an std::vector and we need plain C because the above code will be compiled and executed by the actual GPU. But we're getting there. Please note that a better way than an array with predefined length would be to malloc the necessary space in memory every time, but that is for some other time.  

We build our basic structures for the Node and the Layer so it is time to program the actual Network, which is simply a stack of layers.

```c
h_netVec = newNetVec;

//input layer
Layer *inputLayer = layer(h_netVec[0], 0);
h_layers.push_back(*inputLayer);

///Create the other layers
for (unsigned int i = 1; i <h_netVec.size(); i++)
{
	Layer *hidlayer = layer(h_netVec[i], h_netVec[i - 1]);
	h_layers.push_back(*hidlayer);

}

```

There it is. Our simple Neural network written in C++. In fact, it is nothing more than a vector of layers, with each layer being a vector of Nodes. You may think that our job is done here. Haha! We are not even close. We have to train our network with actual data. This is the time where OpenCL is coming into play.

Those vectors can not be accesed by the GPU so we have to transform them into another structure called Buffer, a basic element of OpenCL. But the logic is exactly the same as before.

```c
d_InputBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*inpdim*inpdim);


tempbuf = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(Node)*h_layers[0].numOfNodes);
(OpenCL::clqueue).enqueueWriteBuffer(tempbuf,CL_TRUE,0,sizeof(Node)*h_layers[0].numOfNodes,h_layers[0].nodes);
d_layersBuffers.push_back(tempbuf);

for (int i = 1; i<h_layers.size(); i++) {
	tempbuf = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(Node)*h_layers[i].numOfNodes);
	(OpenCL::clqueue).enqueueWriteBuffer(tempbuf, CL_TRUE,0, sizeof(Node)*h_layers[i].numOfNodes, h_layers[i].nodes);
	d_layersBuffers.push_back(tempbuf);

}

```

Don't get confused by all those "cl::" , "clqueue" and "context". Those are OpenCL stuff. The logic remains intangible.

Before we close the first part ,we have to one more thing. We have to define the OpenCL Kernels. The kernels are the acual code that is executed by the GPU.
We need 3 kernels in total:
* One for the forward propagation
* One for the backward propagation in the output layer
* One for the backward in the hiddens layer

```c
compoutKern = cl::Kernel(OpenCL::clprogram, "compout");
backpropoutKern = cl::Kernel(OpenCL::clprogram, "backpropout");
bakckprophidKern = cl::Kernel(OpenCL::clprogram, "backprophid");

```
I think that is a lot to digest so i am stopping for now. In the next part we are going to write the actual kernel code and finally train our Neural Network. Stay tuned..
