# Chapter 15 NLP-Pretraining

## 15.1 Word Embedding (word2vec)

* Word Embedding: mapping words to real vectors

### One-Hot Vectors are a bad choice

* cannot accurately express the similarity like cosine similarity

* cosine similarity: 
  
  * $\frac {x^{\top}y} {||x||||y||} \in [-1, 1]$



### Self-Supervised word2vec

* word2vec
  
  * skip-gram
  
  * continuous bag of words (CBOW)

### The skip-gram model

* The likelihood function of skip-gram model is the probablity of generating all context words given any center word



### The continuous Bag of words Model

* it assumes that a center word is generated based on its surrounding context words in the text sequence



## 15.2 Approximate Training

* negative sampling 

* hierarchical softmax

### Negative Sampling

* $P(D=1|w_c, w_o) = \sigma(u_o^\top v_c)$

* $\sigma(x) = \frac 1 {1+exp(-x)}$

### Hierarchical Softmax



### Summary:

* Negative sampling constructs the loss function by considering mutually independent events that involve both positive and negative examples. The computational cost for training is linearly dependent on the **number of noise words** at each step.

* Hierarchical softmax constructs the loss function using the path from the root node to the leaf node in the binary tree. The computational cost for training is dependent on the **logarithm of the dictionary size** at each step.


