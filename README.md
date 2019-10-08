# qonet

# Dependencies
qonet is written in Python3 and requires Torch, Numpy, Torchvision.

# Tests
## parameterizations of x^2
We explore whether or not two difference parameterizations for approximating x^2 make a big difference or not. 
The aspects considered:
* Round off error when run in half percision
* Differences as initializations for training neural networks

## polynomial initialization for classification
Does polynoial initialization help in the classificaiton problem?
* Taking a pretrained AlexNet image classifier, we replace the fully connected layers with a polynoimal initialized network.
* The performance of the original network, the AlexNet trained from scratch, the AlexNet with polynomial architecture networks with random initialization, the AlexNet with polynomial architecture layers with polynomial initialization.

