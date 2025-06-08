# Modern CNN

## 8.1 Deep Convolutional Neural Networks (AlexNet)

### Representation Learning

* AlexNet can be trained on much more data and on much faster GPUs then LeNet about 20 years before. 

### Missing Ingredient: Data, Hardware

### AlexNet

* 8 layers: 5 convolutional layers, 2 fully connected hidden layers and 1 fully connected output layer. 

* ReLU rather than sigmoid in AlexNet
  
  * ReLU has a simpler computation than sigmoid
  
  * easier for diff para init, as the gardient is almost 0 when output of sigmoid is close to 0 or 1.  (improve the training efficiency)

* model complexity of fully connected layer by dropout 
  
  * LeNet only uses weigh decay

* Data augmentation
  
  * more robust and reduces overfitting

## 8.2 Networks Using Blocks (VGG)

### VGG Blocks:

* Visual Geometry Group (VGG)
* building block of CNN: 
  * conv layer with padding to maintain the resolution
  * nonlinearity like ReLU
  * pooling layer like max-pooling to reduce resolution
  * issue: spatial resolution decreases quite rapidly
    
       

### VGG Network

* VGG consists of blocks of layers while AlexNet's layers are all designed individually

* "From layers to block of layers"
