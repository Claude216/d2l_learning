# Modern RNNs

## 10.1 Long Short-Term Memory (LSTM)

* Intuition: 
  
  * Simple RNN have long-term memory in the form of weights, and these weights change slowly during training, enchoding general knowledge abotu the data;
  
  * They also have short memory in the form of ephemeral activations, which pass from each node to successive nodes.

* LSTM introduces an intermediate type of storage via the memory cell. A memory cell is a composite unit, built from simpler nodes in a specific connectivity pattern, with the novel inclusion of multiplicative nodes. 
  
  

### Gated Memory Cell

* Each mem cell equipped with an *internal state* and a number of multiplicative gates that determine whether :
  
  * a given input should impact the internal state (the input gate)
  
  * the internal state should be flushed to 0 (the forget gate)
  
  * the internal state of a given neuron should be allowed to impact the cell's output (the output gate)
    
    
    
    

#### Gated Hidden State

* We have dedicated mechanisms for when a hidden state should be updated and also for when it should be reset. 

* This mechanisms are learned
  
  * If the first token is of great importance we will learn NOT to update the hidden state after the first observation
  
  * skip irrelevant temporary observations
  
  * learn to reset the latent state whenever needed

#### Input Gate, Forget Gate, Output Gate

* 3 fully connected layers with sigmoid activation functions compute the values of the 3 gates. 

* An input node is also required 

* Input gate: determines how much of the input node's value should be added to the current mem cell internal state.

* Forget gate: determines whether to keep current value of hte memory or flush it

* Output gate: determines whether the mem cell should influence the output at the current time step. 
  
  * $I_t = \sigma(X_tW_{xi} + H_{t-1}W_{hi} +b_i$
  
  * $F_t = \sigma(X_tW_{xf} + H_{t-1}W_{hf} +b_f$
  
  * $O_t = \sigma(X_tW_{xo} + H_{t-1}W_{ho} +b_o$
  
  * $W_{xi}, W_{xf}, W_{xo} \in \R^{d \times h}$
  
  * $W_{hi}, W_{hf}, W_{ho} \in \R^{h \times h}$
  
  * $b_i, b_f, b_o \in \R^{1 \times h}$

#### Input Node

* $\tilde C_t = tanh(X_tW_{xc} +H_{t-1}W_{hc} +b_c)$

#### Memory Cell Internal State

* Update equation: $C_t = F_t \odot C_{t-1} + I_t \odot \tilde C_t$

#### Hidden State:

* Update equation: $H_t = O_t \odot tanh(C_t)$
  
  
  
  
  
  
  
  

## 10.2 Gated Recurrent Units (GRU)

### Reset Gate and Update Gate

* The 3 gates of LSTM are replaced by two: the reset gate and the update gate.

* Reset gate: how much of previous state we might still want to remember

* Update gate: how much of the new state is just a copy of the old one. 

* $R_t = \sigma (X_tW_{xr} + H_{t-1}W_{hr} + b_r$

* $Z_t = \sigma (X_tW_{xz} + H_{t-1}W_{hz} + b_z$

* $W_{xr}, W_{xz} \in \R^{d \times h}$

* $W_{hr}, W_{hz} \in \R^{h \times h}$

* $b_r, b_z \in \R^{1 \times h}$

### Candidate Hidden State

* candidate hidden state: $\tilde H_t = tanh(X_tW_{xh} + (R_t \odot H_{t-1})W_{hh} + b_h)$

* tanh is the activation function

* The influence of the previous states can now be reduced with the elementwise multiplication of $R_t$ and $H_{t-1}$ 
  
  * Entries in teh reset gate $R_t$ are close to 1 $\rarr$ vanilla RNN
  
  * .... clase to 0, the candidate hidden state is the result of an MLP with $X_t$ as input. Any pre-existing hidden state is thus reset to defaults. 
    
    

### Hidden State

* Update equation for GRU: $H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde H_t$
  
  

## Deep RNNs

* The hidden state of the $l^{th}$ hidden layer that uses the activation function $\phi_l(H_t^{(l-1)}W_{xh}^{(l)} + H_{t-1}^{(l)}W_{hh}^{(l)} + b_h^{(l)}$

* The output layer at the end: $O_t = H_t^{(L)}W_{hq} + b_q$

* num of hidden layers L and num of hidden units h are hyperparameters that we can tune
  
  * common h: (64, 2056)
  
  * common depths L: (1, 8)
    
    

## 10.4 Bidirectional Recurrent Neural Networks

* Forward hidden state update function: $\overrightarrow H_t = \phi(X_tW_{xh}^{(f)} + \overrightarrow H_{t-1}W_{hh}^{(f)} + b_h^{(f)})$

* Backward hidden state update function: $\overleftarrow H_t = \phi(X_tW_{xh}^{(b)} + \overleftarrow H_{t+1}W_{hh}^{(b)} + b_h^{(b)})$

* Output layer: $O_t = H_tW_{hq} + b_{q}$

* Bidirectional RNNs are mostly useful for sequence encoding and the estimation of observations given bidirectional context. 
  
  * But costly to train due to long gradient chains
    
    
    
    

## 10.5 Machine Translation and the Dataset

### Tokenization

* For machine translation we prefer word-level tonization rather than a character-level one
  
  

### Summary

* Machine Translation: the task of *automatically mapping from a sequence representting a string of text in a source language to a string representing a plausible translation in a  target language.* 

* By truncate and pad text sequences so that all of them will have the same length to be loaded in minibatches. 
  
  
  
  

## 10.6 The Encoder-Decoder Architecture

* [Input] $\rarr$ [Encoder] $\rarr$ [State] $\rarr$ [Decoder] $\rarr$ [Output]

### Encoder

### Decoder



## 10.7 Seq2Seq Learning for Machine Translation

### Teacher Forcing

* The original target seq is fed in to the decoder as input. 

### Encoder

* transformation of the RNN's recurrent layer: 
  
  * $h_t = f(x_t, h_{t-1}$

* encoder transformation: 
  
  * $c = q(h_1, ..., h_T)$

### Decoder

* transformation of decoder's hidden layer: 
  
  * $s_{t^{'}} = g(y_{t^{'}-1}, c, s_{t^{'}-1})$



### Evaluation

* BLEU (bilingual evaluation understudy)
  
  * $exp(min(0, 1 - \frac {{len}_{label}} {{len}_{pred}})) \sum^k_{n=1}p_n^{\frac 1 {2^n}}$
  
  * k is the longest n-gram for matching



## 10.8 Beam Search

* 
