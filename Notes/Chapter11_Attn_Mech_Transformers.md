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
  
  

## 11.5 Multi-Head Attention

* Q,K, and V can be transformed with h independentaly learned linear projections. Then these h projected queries, keys, and values are fed into attention pooling in parallel. 

* In the end, h attention pooling outputs are concatenated and transformed with another learned linear projection to produce the final output. 

* The multi-head attention

### Model

* $h_i = f(W_i^{(q)}q, W_i^{(k)}k, W_i^{(v)}v) \in \R^{p_v}$

* these w are learnable parameters and f is attention pooling

* The multi-head attention output: $W_o \begin{bmatrix} h_1 \\ \vdots \\ h_h \end{bmatrix} \in \mathbb{R}^{p_o}$
  
  

## 11.6 Self-Attention and Positional Encoding

### Self-Attention



### Comparing CNNs, RNNs, and Self-Attention

* CNN: Given a sequenceo f length n, and a conv layer whose kernel size is k, and num of input and output channels are both d, the computational complexity of the conv layer is $O(knd)^2$

* RNN: multiplication of the $d \times d$ weight matrix and the hidden state is $O(d^2)$. There are $O(n)$ sequential operations that cannot be parallelized and the maximum path length is also $O(n)$

* Self-Attn: q, k, v are all $n \times d$ matrices, so the complexity is $O(n^2d)$. Computation can be parallel with $O(1)$ sequential operations and max path length is also $O(1)$ as teh token is connected to any other token via self-attention. 
  
  
  
  
  
  

## 11.7 The Transformer Architecture

* parallel computation and the shortest maximum path length. 

### Model

* Transformer encoder: a stack of multiple identical layers, each layer has two sublayers
  
  * Multi-head self-attention pooling
  
  * Positionwise feed-forward network. 
  
  * q, k, v all from the outputs of the previous encoder layer

* Transformer decoder: a stck of multiple idential layers with residual connections and layer normalizations
  
  * layer 1 same as encoder
  
  * layer 2 same as encoder
  
  * encoder-decoder attention: 
    
    * queries from the outputs of decoder's self-attention sublayer
    
    * keys and avalues from the Transformer encoder outputs.

### Positionwise Feed-Forward Networks

* Positionwise feed-forwawrd network transforms the representation at all the sequence positions using the same MLP.
  
  

### Residual Connection and Layer Normalization

* The residual connection requires the two inputs are of the same shape so that the output tensor also has the same shape after the addition operation.
  
  

### Encoder:

* Multi-head attention

* addnorm

* positionwise ffn

* addnorm

* output of Transformer encoder output: (batch size, num of time steps, num_hiddens)
  
  

### Decoder:

* decoder self-attention

* encoder-decoder attention

* positionwise ffn





## 11.8 Transformers for Vision

* Cordonnier, J.-B., Loukas, A., & Jaggi, M. (2020). On the relationship between self-attention and convolutional layers. _International Conference on Learning Representations_.

* This paper proved that self-attention can learn to behave similarly to convolution. 

* Vision Transofrmers (ViTs) extract patches from images and feed them into a Transformer encoder to obtain a global representation. 

### Model

* An input image with height h, width w, and c  channels, patch height and width both as p, then the image is split into a sequence of $m = hw/p^2$ patches, and each pathc is flattened to a vector of length $cp^2$. 



### Patch Embedding

* split an image into patches and linearly projecting these flattened patches.

### Vision Transformer Encoder

* Diff from positionwise FFN of original Transformer encoder:
  
  * activation func: Gaussian error linear unit (GELU)
    
    * a smoother ReLU
  
  * dropout is applied to the output of each fully connected layer in the MLP for regularization

* Pre-normalization design: normalization is applied before multi-head attention or the MLP

* post-normalization: normalization is placed after residual connections

* pre-norm is more effective / efficient traning for transformers



## 11.9 Large-Scale Pretraining with Transformers

* 3 diff modes: 
  
  * encoder-only
  
  * encoder-decoder
  
  * decoder-only

### Encoder-Only

* A sequence of input tokens is converted into the same number of representations that can be further projected into output

* BERT (Bidirectional Encoder Representations from Transformers)

#### Pretraining BERT

* Pretrained on text sequences using masked language modeling:
  
  * input text with randomly masked tokens is fed into a Transformer encoder to predict the masked tokens. 
  
  * no constraint in the attention pattern: all token can attend to each other $\rarr$ "bidirectional encoder"

* Large-scale text data can be utilized without manual labeling

#### Fine-Tuning BERT

* Loss for predicting whether one sentence immediately follows the other. 



### Encoder-Decoder

* The encoder-only mode cannot generate a sequence of arbitrary length as in machine translation. 
  
  * for conditioning on encoder output, encoder-decoder cross-attention (MHA of decoder) allows target tokens to attend to all input tokens
  
  * conditioning on decoder output is achieved by "causal attention" (has little connection to the proper study of causality) pattern, where any target token can only attend to past and present tokens in the target sequence

* BART and T5



#### Pretraining T5

* It unifies many tasks as the same text2text problem



#### Fine-Tuning T5

* Diff from BERT
  
  * T5 input includes task descriptions
  
  * T5 can generate sequences with arbitrary length with its Transformer decoder
  
  * No additional layers are required

### Decoder-Only

* It removes the entire encoder and the decoder sublayer with the encoder-decoder cross attention. 

#### GPT and GPT-2

* The attention pattern in Transformer decoder enforces that each token can only attend to its past tokens

#### GPT-3 and Beyond

* A pretrained language model can be trained to generate a text sequence conditional on some prefix text sequence.
  
  * A model may generate the output as a sequence without parameter update, conditional on an input sequnece with the task description, task-specific input-output examples, and prompt. ==> in-context learning
  
  * zero-shot, one-shot, and few-shot.

### Scalability

### Large Language Models

* 
