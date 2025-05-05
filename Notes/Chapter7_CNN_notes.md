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
