# Recurrent Neural Networks



## Working with Sequences

* The evolution

* Seq2seq tasks: 
  
  * aligned: input at each time step aligns with a corresponding target
  
  * unaligned: input and target do not necessarily exhibit a step-for-step correspondence 

* Sequence modeling: given a collection of sequences, estimate the probability mass function that tells us how likely we are to see any given sequence. 

### Autogressive Models

* models regress the value of a signal on the previous values of that same signal. 

* $x_t$ is the input, $\hat x_t$ is the output, and $h_t$ is the intemediate layer (the hidden state)

* For a latent autoregressive model: 
  
  * $\hat x_t = P(x_t | h_t)$
  
  * $h_t = g(h_{t-1}, x_{t-1}$ 

### Sequence Models

#### Markov Models:

#### The order of decoding

* factorizing text in the same direction we read it (left to right, or, beginning to the end)





## 9.2 Converting Raw text into Sequence Data

* Typical Preprocessing pipeline for converting raw text into sequences: 
  
  * Load text as strings into mem
  
  * Split strings into tokens (words or characters)
  
  * Build a vocabulary dictionary to associate each vocabulary element with a numerical index
  
  * Convert the text into sequences of numerical indices.
    
     

### Reading the Dataset

### Tokenization

### Vocabulary

* associate each distinct token value with a unique index.
  
  * First, determine the set of unique tokens in our training corpus. 
  
  * Then assign a numerical index to each unique token
    
    * Rare vocabulary elements are often dropped for convenience. 

### Exploratory Language Statistics

* Zipf's Law: 
  
  * The frequency $n_i$ of the $i^{th}$ most frequent word is: $n_i \propto \frac {1} {i^\alpha}$ which is equivalent to $log {n_i} = -\alpha logi + c$
  
  * $\alpha$ is the exponent that characterizes the distribution and $c$ is a constant.

### Summary:

* preprocess text: 
  
  * (i) split text into tokens
  
  * (ii) build vocabulary to map token strings to numerical indices
  
  * (iii) convert text data into token indices for models to manipulate
