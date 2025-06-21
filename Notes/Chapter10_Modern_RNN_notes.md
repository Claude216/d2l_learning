# Modern RNNs

# Long Short-Term Memory (LSTM)

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

* 
