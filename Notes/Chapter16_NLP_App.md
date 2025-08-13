# NLP: Applications

## 16.1 Sentiment Analysis and the Dataset

* Stanford's large moview review dataset for sentiment analysis

### Reading the Dataset



## 16.2 Sentiment Analysis: Using Recurrent Neural Networks

* represent each token using GloVe model, and feed them into a multi-layer bidirectional RNN to obtain the text sequence representation. 
  
  

### Representing Single Text with RNNs





## 16.3 Sentiment Analysis: Using Convolutional Neural Networks

* textCNN

### One-Dimensional Conv

### Max-Over-Time Pooling

* diff num of time steps at diff channels

### The textCNN Model

* tansforms individual token representations into downstream application outputs using one-dimensional convolutional layers and max-over-time pooling layers. 
  
  

## 16.4 Natural Language Inference and the Dataset

* reason over pairs of text sequences

### Natural Language Inference

* where a hypothesis can be inferred from a premise.

* 3 types of relationships: 
  
  * Entailment: hypothesis can be inferred from the premise
  
  * Contradiction: the negation of the hypothesis can be inferred from the premise
  
  * Neutral: all the other cases

### SNLI

## 16.5 NL Inference: Using Attention

* decomposable attention: 
  
  * attending
  
  * comparing
  
  * aggregating

* align tokens in one text sequence to every token in the other and vice versa. 

* decomposition trick leads to a linear complexity than quadratic complexity 

* 
