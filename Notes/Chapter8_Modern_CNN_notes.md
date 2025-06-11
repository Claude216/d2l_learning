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
  
  

## 8.3 Network in Network (NIN)

### NiN Blocks

* No giant fully connected layers 

* Instead, global avg pooling to aggragate across all image locations after the last stage of the network body.
  
  * The avg operation did NOT harm accuracy. 





## 8.4 Multi-Branch Networks (GoogLeNet)

### Inception Blocks

* Explore the image in a variey of filter sizes. 

### GoogLeNet:

* This model is computationally complex


