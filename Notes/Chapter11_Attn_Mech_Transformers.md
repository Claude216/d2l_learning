# Attention Mechanisms and Transformers

## 11.1 Queries, Keys, and Values

* queries q operate on (k, v) pairs

* $D  \stackrel{def}{=} \{{(k_1, v_1), ...(k_m, v_m)}\}$

* attention over D: $Attention(q, D) \stackrel{def}{=} \sum^m_{i=1} \alpha(q, k_i)v_i$
  
  * attention pooling
  
  * $\alpha(q, k_i) \in \R$ are scaler attention weights
  
  * sepacial cases: 
    
    * Nonnegative 
    
    * Weights form a convex combination. i.e., $\sum_i \alpha(q, k_i)=1$ and $ \alpha(q, k_i) \geq 0$; and this is the most common cases
    
    * Exactly one of the weights is 1, while all others are 0. Akin to a traditional database query
    
    * All weights are equal. Avg pooling in deep learning
  
  * Normalizing to ensure the sum up to 1
    
    * $\alpha(q, k_i) = \frac {\alpha(q, k_i)}{\sum_j \alpha(q, k_i)}$
  
  * To ensure nonnegative, we can resort to exponentiation. 
    
    * $\alpha(q, k_i) = \frac {exp(a(q, k_i))} {\sum_j exp(a(q, k_j))}$

### Visualization

### Summary

* Attention Mechanism provides a differentiable means of control by which a neural network can select elements from a set and to construct an associated weighted sum over representations. 





## 11.2 Attention Pooling by Similarity

* common kernels: 
  
  * Gaussian: $\alpha(q, k) = exp(- \frac 12 ||q-k||^2)$
  
  * Boxcar: $\alpha(q, k) = 1 if ||q-k|| \leq 1$
  
  * Epanaechikov: $\alpha(q, k) = max(0, 1-||q-k||)$

* equation for regression and classification: 
  
  * $f(q) = \sum_i v_i \frac {\alpha(q, k_i)} {\sum_j \alpha(q, k_j)}$

* use one-hot-encoding of $y_i$ to obtain $v_i$ 



### Kernels and Data

* Diff kernels correspond to different notions of range and smoothness

### Attention Pooling via Nadaraya-Watson Regression

### Adapting Attention Pooling

* replace Gaussian Kernel with one of a different width

* $\alpha(q,k) = exp(- \frac 1{2\sigma^2} ||q-k||^2)$
  
  * $\sigma^2$ determines the width of the kernel. 








