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
