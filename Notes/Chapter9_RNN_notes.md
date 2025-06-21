# Recurrent Neural Networks



## 9.1 Working with Sequences

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
    
    

## 9.3 Language Models

### Learning Language Models

#### Markov Models and n-grams

* probability formulae that involve: 
  
  * one $\rarr$ unigram
  
  * two $\rarr$ bigram
  
  * three $\rarr$ trigram

#### World Frequency

* $\hat P (Y | X) = \frac {n(X, Y)} {n(X)}$

* $n(X, Y)$: num of occurrences of consecutive word pair XY

* $n(X) $: num of occurrences of singletons X

#### Laplace smoothing

* many n-grams occur very rarely, making laplace smoothing rather unsuitable for language modeling.

* we need to store all counts

* this entirely ignoires the meaning of the words. 
  
  

### Perplexity:

* how to measure the quality of the language model? 
  
  * by computing the likelihood of the sequence. 
  
  * by the cross-entropy loss averaged over all the n tokens of a sequence
  
  * perplexity: 
    
    * $exp(- \frac 1 n \sum^n_{t=1} logP(x_t|x_{t-1}, ..., x_1))$
      
      
      
      

### Partitioning Sequences

* how to read minibatches of input sequences and target sequences at random.
  
  * partition the corpus (contains token indices) into subsequences, each has n tokens. 
  
  * at the begining of each epoch, discard the first d tokens where d is uniformly sampled at random. The rest of hte sequence is then partitioned into $m = \lfloor (T-d) / n\rfloor$  subseuqnces. 
    
    
    
    
    
    

## 9.4 Recurrent Neural Networks

* RNN: neural networks with hidden states
  
  

### Neural Networks without hidden States

* MLP
  
  

### Recurrent Neural Networks with Hidden States

* $H_t = \phi(X_tW_{xh} + H_{t-1}W_{hh} + b_h$ 

* NN with hidden states based on recurrent computation are named RNN

* Layers perform the computuation of this formula are called recurrent layers

* With recurrent computation the number of RNNmodel parameters does not grow as the number of time steps increases. 
  
  
  
  

## 9.5 RNN implementation from Scratch

### RNN model

### RNN-Based Language Model

#### One-Hot Encoding:

* e.g., label is 1 out of [0, 1, 2] $\rarr$ [0, 1, 0]

#### Transforming RNN Outputs

* Language model uses a fully connected output layer to transform RNN outputs into token pred at each *time step*
  
  

### Gradient Clipping

* RNNs sufer from exploding gradients

* so we clip the gradients to force the resulting gradients to take smaller values. 

* Lipschitz constinuous with constant L: 
  
  * $|f(x) - f(y)| \le L||x - y||$ 

* one way to limite the size of $L\eta ||g||$ is to shrink the learning rate $\eta$ to tiny values.
  
  

### Decoding:

* Warm-up
  
  * Looping through the characters in prefix, keep passing the hidden state to the next time step but do not generate any output.
    
    
    
    

## 9.6 Concise Implementation of Recurrent Neural Networks





## 9.7 Backpropagation Through Time

### Analysis of Gradients in RNNs



### Backpropagation Through Time in Detail

* 
