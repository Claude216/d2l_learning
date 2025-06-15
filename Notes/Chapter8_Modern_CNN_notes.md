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
  
  
  
  

## 8.5 Batch Normalization

* A popular and effective tech for accelerating the convergence of deep networks.

### Traning Deep Networks

* Why we want batch normalization? 
  
  * The standardization plays nicely with the optimizers since it puts the parameters a priori on a similar scale. 
  
  * variables in intermediate layers may take values with widely varying magnitudes: the drift in the distribution of such variables could hamper the convergence of the network.
  
  * Deeper NN are complex and tend to be more liable to overfitting: regularization becomes more critical.
    
    * noise injection

* Batch normalization: preprocessing, numerical stability and regularization. 

* $BN(x) = \gamma \odot \frac {x - \hat{\mu}} {\hat{\sigma}} + \beta$

* BN layers function differently in *training mode* (normalizing by minibatch statistics) than in *prediction mode* (normalizing by dataset statistics). 

### Batch Normalization Layers

#### Fully Connected Layers

* in original paper, BN was before the nonlinear activation function

* Later applications epxerimented BN after the activation functions. 

#### Convolutional Layers

#### Layer Normalization:

* Adv: 
  
  * it prevents divergence; the output of the layer normalization is scale independent if ignoring $\epsilon$
  
  * it doesn't depend on the minibatch size. 
    
    * it's simply a deterministic transformation that standardizes the activations to a given scale. 

#### Batch Normalization During Prediction

* noise in sample mean and sample variatnce is no longer desirable once model trained. 

* After training, use the entire dataset to compute stable estimates of the variable statistics and then fix them at prediction time. 
  
  ### 

### Discussion

* why is BN effective? 
  
  * reducing internal covariate shift.

* some takeaways: 
  
  * BN adjusts the intermediate output of networks by utilizing the mean and standard deviation of the minibatch --> the values of the intermediate output in each layer throughout the neural network are more stable
  
  * BN is slightly different for fully connected layers than for convolutional layers. For Conv layers, LN can be used as an alternative
  
  * Like a dropout layer, BN layers have diff behaviors in training than in prediction mode
  
  * BN is useful for regularization and improving convergence in optimization. However, reducing internal covariate shift seems not  to be a valid explanation.
  
  * For models that are less sensitive to input perturbations, consider removing BN.
    
    
    
    

## 8.6 Residual Networks (ResNet) and ResNeXt

### Residual Blocks

* residual mapping $g(x) = f(x) - x$

* a shortcut for the input to forward propagate faster
  
  
  
  
  
  

## 8.7 Densely Connected Networks (DenseNet)

### From ResNet to DenseNet

* use concatenation [ , ] rather than + 
  
  * ResNet: $x \rarr x + f_1(x) + f_2(f_1(x)) + ...$
  
  * DenseNet: $ x \rarr [x, f_1(x), f_2([x, f_1(x), ...]$

* "Dense"Net: the final layer of such concatenation is densely connected to ALL previous layers. 

* Main compounents: *dense blocks* and *transition layers*
  
  * dense blocks define how the inputs and outputs are concatenated
  
  * transition layers control the number of channels to avoid becoming too large as the expansion of denseNet can be quite high-dimensional
    
    

### Dense Blocks

* Modified "batch normalization", "activation", and "convolution" of ResNet. 
  
  * multiple conv blocks with same number of output channels. 
  
  * concatenate the input and output of each conv block on the channel dimension in forward propagation. 
  
  * output channel # = input channels + conv channels + output channels 
  
  * Num of conv block channels controls the growth in the num of output channels relative to the number of input channels $\rarr $ growth rate is another name of num of channels for conv layers

### Transition Layers

* Each dense block will increase the number of channels
  
  * Complexity increases as adding too many of them

* Transition layer
  
  * reduces the num of channels by using a $1 \times 1$ conv. 
  
  * halves the height and width via avg pooling with a stride of 2

### Discussion:

* DenseNet may require more GPU memory when training





## Designing Convolution Network Architectures


