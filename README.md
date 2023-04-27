# CSC3022F : Artificial Neural Networks
## Machine Learning Assignment 1
Machine Learning - Designing nueral networks to successfully classify objects of images in the CIFAR10 dataset by training the network and testing to achieve a degree level of accuracy to classify a component. Using python pytorch and torchvision libraries

## Project Description
- MLP - A standard feed-forward Multi-Layer Perceptron that has 3 hidden layers and perfom a batch normalisation from the input and hidden layers that will divide each feature to its standard deviation as a batch size sampled at 128, also with a dropout of 20% and perform a ReLu activation function to model the weight of each nueron. The network uses a stochastic gradient descent with a low learning rate of 0.1 to decrease the error rate and a momentum of 0.9 that determine how far to push the weights against negative gradient to decrease error rate. And also use a learning rate decay to decrease the learning rate after 10 epochs of 15 epochs. The loss function used is Cross Entropy loss.

- CNN - The Convolutional Neural Network is using convolutional layers to improve Artificial Neural Networks classification accuracy. The file implements a Network of batch size 16 with 2 convolutional layers with filter size 5x5 to extract a feature map and each is passed to the ReLu activation function after that follow a Max pooling layer with filter size 2x2 and a stride of 2 on each feature to produce pooled maps. The first convolutonal has input channel of 3 and output channel of 6, while the second convolutional layer has a input channel of 6 and output of 16. The pooled map is then flattened to produce a 1d image as we working with image classification and passed to a 3 Fully connected layer where the last one is the output. The first two Fully connected layer do a batch normalisation from their output to devide each feature weight to its standard deviation and perform a ReLu  activation function. The CNN uses a stochastic gradient descent with a low learning rate of 0.1 to decrease the error rate and a momentum of 0.6 that determine how to counter the weights against the negative gradient to decrease the error rate using these parameter as optimizer which is for updating the weights and does a learning rate decay from the optimiser to decrease the learning rate by 0.1 after 5 epochs out of a total of 10. The loss function at the output layer used is a Cross Entropy loss.

## How to execute Project

By using a makefile to run following commands

To delete venv folder 
```bash
make clean
```
To setup a python virtual environment.
```bash
make
```
To execute project using make file argumented execution for Multi-Layer perceptron
```bash
make runMLP
```
To execute project using make file argumented execution for part 2 LeNet5 - Convolutional Neural Network
```bash
make runCNN
```
To execute project using make file argumented execution for Residual connections
```bash
make runRES
```

