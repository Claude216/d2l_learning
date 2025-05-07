# Convolutaional Neural Networks

## 7.1 From Fully Connected Layers to Convolutions (05/03/2025)

### Principles:

- Translation invariance: network should respond similarly to the same patch, regardless of where it appears in the image
  
  - A shift in the input should simply lead to a shift in the hidden representation

- Locality principle: The earliest layers of the network should focus on local regions, without regard for the contents of the image in distant regions.

### Convolutions:

- $(f * g)(\bold{x}) = \int f(\bold{z})g(\bold{x} - \bold{z})d\bold{z}$ 
  
  

## 7.2 Convolutions for Images (05/05/2025)

* Kernel for Croess-Correlation Operation
  
  * for kernel $k_h \times k_w$,
  
  * input $n_h \times n_w$
  
  * output size: $(n_h - k_h + 1) \times (n_w - k_w + 1)$

* Feature Map and Receptive Field





## 7.3 Padding and Stride

#### Padding:

- padding: $p_h$ rows, $p_w$ columns
  
  - output shape: $(n_h - k_h + p_h + 1) \times (n_w - k_w + p_w + 1)$

- Commly use convolution kernels with **odd** height and width values, (e.g., 1, 3, 5, 7)

- Setting the padding size when the height and width of convolution kernel are not the same can control the output and input with the same sizes. 
  
  ### 

#### Stride:

* The step each time we move the conv kernel

* for stride $s_h$ and $s_w$: 
  
  * output shape: $\left\lfloor (n_h - k_h + p_h + s_h)/s_h \right\rfloor \times \left\lfloor (n_w - k_w + p_w + s_w)/s_w \right\rfloor$ 

 


