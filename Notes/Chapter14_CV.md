# Chapter 14 Computer Vision

## 14.1 Image Augmentation

### Common Image Augmentation Methods

#### Flipping and Cropping

* Flipping the image left and right 
  
  * easiest and most common
  
  * horizontal and vertical flipping

* Crop area of image and scale the rest

#### Changing Colors

* brightness

* contrast

* saturation

* hue

#### Combining Multiple Image Augmentation Methods

* Example
  
  * ```python
    augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
    apply(img, augs)
    ```

### Training with Image Augmentation



## 14.2 Fine-Tuning

* Collecting more data

* transfer learning 

### Steps

* 1. Pretrain a NN model, the source model on a source dataset
  
  2. Create a new NN model, the target model
     
     * copies all model designs and parameters on the source model EXCEPT the output layer
  
  3. Add an output layer to the target model, whose num of outputs is the num of categories in the target dataset. Then randomly initialize the model parameters of this layer
  
  4. Train the target model on the target dataset. The output layer will be trained from scratch, while the parameters of all the other layers are fine-tuned based on the parameters of the source model.

### Hot Dog Rec

### Summary

* Transfer learning transfers knowledge learned from the source dataset to the target dataset. Fine-tuning is a common tech for transfer learning

* The target model copies all the model designs with their parameters from the source model except the output layer, and fine-tunes these parameters based on the target dataset. But the output layer of target model needs to be trained from scratch

* Fine-tuning parameters uses a smaller learning rate, while training the output layer from scratch can use a larger learning rate
  
  

## 14.3 Object Detection and Bounding Boxes

* not just categories but also positions

### Bounding Boxes

* convert between two commonly used bounding box representations
  
  

## 14.4 Anchor Boxes

### Generating Multiple Anchor Boxes

* when the center position is given, an anchor box with know width and height is determined

### Intersection Over Union (IoU)

* $J(A, B) = {|A \cap B| \over |A \cup B|}$

### Labeling Anchor Boxes in Training Data

* calss and offset labels for each anchor box

* labeled location and class of its assigned ground-truth bounding box that is closest to the anchor box. 

#### Assigning Ground-Truth Bounding Boxes to Anchor Boxes

#### Labeling Classes and Offsets



### Predicting Bounding Boxes with Non-Maximum Suppression (NMS)

* work flow
  
  * for a predicted bounding box B, the objection detection model calculates the predicted likelihood for each class
  
  * p is the largest predicted likelihood, and refer it as the confidence (score) of the predicted bounding box B. 
  
  * all the non-bakcground bounding boxes are sorted by confidence in descending order to generate a list L
  
  * manipulate L in the steps:
    
    * select the bounding box B1 with the highest confidence as a basis and remove all non-basis predicted bounding boxes whose IoU with B1 exceeds a predefined threshold $\epsilon$ from L. L keeps the box with highest confidence but drops others that are too similar to it.
    
    * Select the B2 with the second highest confidence from L as another basis, also remove those non-basis boxes whose IoU with B2 exceeds $\epsilon$ from L
    
    * Repeat until all the boaxes in L has been used as a basis; any IoU of pair of boxes below threshold $\epsilon$; thus no pair too similar with each other
    
    * Ouput all the predicted bounding boxes in list L

## 14.5 Multiscale Object Detection

### Multiscale Anchor Boxes

### Multiscale Detection

### Summary

* Generate anchor boxes with different sizes to detect objects with different sizes

* By defining the shape of feature maps, we can determine centers of uniformly sampled anchor boxes on any image. 
  
  

## 14.6 Object Detection Dataset



## 14.7 Single Shot Multibox Detection

### Model

* class prediction and bounding box prediction

#### Class Prediction Layer

* SSD uses channels of convolutional layers to predict classes to reduce model complexity

* there is a one-to-one correspondence between outputs and inputs at the same spatial dimensions of feature maps. 

#### Bounding Box Prediction Layer

* num of output for each anchor box: 4 offsets rather than q+1 classes

#### Concatenating Predictions for Multiple Scales





## 14.8 Region-based CNNs (R-CNNs)

### R-CNNs

* it first extracts many region propasals from the input image, labeling their classes and bounding boxes

* perform forward propagation on each region proposal to extract its features. Then features are used for predicting the class and bounding box of this region proposal



## 14.9 Semantic Segmentation and the dataset

* how to divide an image into regions belonging to different semantic classes. 

### Image Segmentation and Instance Segmentation

* 2 tasks similar to semantic segmentation
  
  * Image segmentation
    
    * divide into several constituent regions
    
    * correlation between pixels in the image
    
    * no need for label information about image pixels during trianing
    
    * cannot guarantee the segmented regions will have the semantics we hope to obtain.
  
  * instance segmentation
    
    * simultaneous detection and segmentation
    
    * how to recognize pixel-level regions of each obj instance 

### The Pascal VOC2012 Semantic Segmentation Dataset



## 14.10 Transposed Convolution

* fractionally-strided convolution

### Basic Operation

* regular conv reduces input elements via the kernel

* transposed conv broadcasts input elementats via the kernel

### Padding, Strides, and Multiple Channels

* regular's padding applied to input

* transposed's padding applied to output

### Connection to Matrix Transposition

* 
