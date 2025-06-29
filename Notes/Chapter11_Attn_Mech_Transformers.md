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
    
    
    
    
    
    

## 11.3 Attention Scoring Functions

* distance functions are slightly more expensive to compute than *dot products*

* Attention Scoring Functions

### Dot Product Attention

* $\alpha(q, k_i) = - \frac 1 2 ||q-k_i||^2 = q^{\intercal}k_i - \frac 1 2 ||k_i||^2 - \frac 1 2||q||^2 $

* The first commonly used attention function in Transformers: 
  
  * $a(q, k_i) = \frac {q^\intercal k_i} {\sqrt{d}}$
  
  * To ensure the variance of the dot product remains 1 regardless of vector length, we use $\frac 1 {\sqrt{d}}$ to rescale the dot product.

* As the attention weights $\alpha$ still need normalizing: 
  
  * $\alpha (q, k_i) = softmax(a(q, k_i)) = \frac {exp(\frac {q^\intercal k_i} {\sqrt{d}})} {\sum_{j=1}exp(\frac {q^\intercal k_i} {\sqrt{d}})}$ 

### Conenvience Functions

### Masked Softmax Operation

* Since we do not want blanks in our attention model we simply need to limit the length of the actual sentence l <= n. $\rarr$ *masked softmax operation*

* The implementation set the values of $v_i$ for $i > l$ to 0, and sets the attention weights to a large negative num, like $-10^6$. 
  
  * $\sum^n_{i=1}\alpha(q, k_i)v_i \rarr \sum^l_{i=1}\alpha(q, k_i)v_i$



#### Batch Matrix Multiplication

* $Q = [Q_1, Q_2, ..., Q_n] \in \R^{n \times a \times b}$

* $K = [K_1, K_2, ..., K_n] \in \R^{n \times b \times c}$

* $BMM(Q, K) = [Q_1K_1, Q_2K_2, ..., Q_nK_n] \in \R^{n \times a \times c}$



### Scaled Dot Product Attention

* $softmax(\frac {QK^\intercal} {\sqrt{d}})V \in \R^{n \times v}$

* Use dropout for model regularization



### Additive Attention

* This make the attention additive and lead to some minor computational savings. 
  
  * $a(q, k) = w_v^{\intercal}tanh(W_qq + W_kk) \in \R$
  
  * Then this term is fed into a softmax to ensure both nonegativity and normalization. 
  
  * THe query and key are concatenated and fed into an MLP with a single hidden layer. 



## The Bahdanau Attention Mechanism

* When predicting a token, if not all the input tokens are relevant, the model aligns only to parts of hte input sequence that are deemed relevant to the current prediction. 



### Defining the decoder with Attention

* state of the decoder is initialized with
  
  * hidden states of the last layer of hte encoder at all time steps
  
  * hidden state of the encoder at all layers at the final time step, which initialize the hidden state of the decoder.
  
  * valid length of the encoder, to exclude the padding tokens in attention pooling. 

### Summary:

* In the RNN encoder-decoder, the Bahdanau attention mechanism treats the decoder hidden state at teh previous time step as the query, and the encoder hidden states at all the time steps as both the keys and values. 


