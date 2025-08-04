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
  
  

## 15.3 The Dataset for Pretraining Word Embeddings

* Replace those appear less than 10 times with "<unk>"

### Subsampling

* for those high-frequency words, we still need to do a subsampling
  
  * some word like "a", "the" appear almost everywhere but with little meanings.
  
  * training with vast amounts of such words is slow.

* Each indexed word $w_i$ in the dataset will be discarded with:
  
  * $P(w_i) = max(1- \sqrt{\frac t {f(w_i)}}, 0)$
  
  * t is a hyperparameter (e.g., $10^{-4}$)
  
  * those with a much higher appearance would be discarded here

### Negative Sampling

* Sample noise words according to a predefined distribution
  
  
  
  

### Loading in Minibatches

* the ith example includes
  
  * a center word
  
  * its $n_i$ context words
  
  * $m_i$ noise words

* For each example we concatenate, pad zeros until the concatenation length reaches. 

* Define a mask variable to excluding paddings in loss calculation. 

* using labels variable to separate context words from noise words 
  
  

## 15.4 Pretraining word2vec

### The Skip-Gram Model

* Using Binary Cross-Entropy Loss as the loss

* Using cosine similarity of word vectors to find the word from the dictionary that are most semantically similar to an input word.
  
  

## 15.5 Word Embedding with Global Vectors (GloVe)

* statistical information in the entire corpus for word embedding 

### Skip-Gram with Global Corpus Statistics

### The GloVe Model

* 3 changes on skip-gram model
  
  * use variables $p'_{ij} = x_{ij}$ and $q'_{ij} = exp(\bold{u}_j^\top \bold{v}_i)$ that are NOT probability distributions and take the logarithm of both, so the squared loss term is $(log \space p'_{ij} - log \space q'_{ij})^2 = (\bold{u}_j^\top \bold{v}_i - log \space x_{ij})^2$
  
  * Add two scalar model parameters for each word $w_i$: the center word bias $b_i$ and the context word bias $c_i$. 
  
  * Replace the weight of each loss term with the weight function $h(x_{ij})$, where $h(x)$ is increasing in the interval of $[0, \space 1]$

* Loss function:
  
  * $\sum_{i \in \it{V}} \sum_{j \in V}{h(x_{ij})(\bold{u}_j^\top \bold{v}_i + b_i + c_j - log \space x_{ij})^2}$
    
    

### Summary

* Skip-gram model can be interpreted using global corpus statistics such as word-word co-occurrence counts

* The cross-entropy loss may not be a good choice for measuring the difference of two probability distributions, especially for a large corpus. GloVe uses squared loss to fit precomputed global corpus statistics. 

* The center word vector and the context word vector are mathematically equivalent for any word in GloVe

* GloVe can be interpreted from the ratio of word-word co-occurrence probabilities. 
  
  
  
  

## 15.6 Subword Embedding

## The fastText Model

* a subword embedding approach. 

* a center word is represented by the sum of its subword vectors

* 3-gram example of "where": "\<wh", "whe", "her", "ere", "re\>" and the subword "\<where>"
  
  

### Byte Pair Encoding

* a statistical analysis of the training dataset to discover common symbols within a word. A greedy approach, it merges the most frequent pair of consecutive symbols. 
  
  

### Summary

* Subword embedding may improve the quality of representations of rare words and out-of dictionary words. 





## 15.7 Word Similarity and Analogy

* 50-dimensional GloVe Embeddings



### Applying Pretrained Word Vectors

#### Word Similarity

* cosine similarities. 

* search for similar words using the pretrained word vectors. j

#### Word Analogy

* apply word cectors to word analogy tasks. 

* Ex:
  
  * For a word analogy: $a:b::c:d$,  $vec(d) = vec(c) + vec(b) - vec(a)$





## 15.8 Bidirectional Encode Representations from Transformers (BERT)

### From Context-Independent to Context-Sensitive

* limitation of context-independent
  
  * same word in diff context contains diff meanings

* Context-sensitive word representations
  
  * TagLM
  
  * CoVe
  
  * ELMo

### From Task-Specific to Task-Agnostic

* GPT: general task agnostic model for context-sensitive representations

* ELMo freezes parameters of the pretrained model, GPT fine-tunes all the parameters in the pretrained Transformer decoder during supervised learning of the downstream task. 

* Autoregressive nature of LM: only looking forward (left-to-right)



### BERT: Combining the Best of Both Worlds

* represent any token based on its bidirectional context.

### Input Representation

* BERT input sequence:
  
  * <cls> 
  
  * <sep>

* One BERT input sequence may include 1 text sequence or 2 text sequences

* Transformer encoder as its bidirectional architecture. 
  
  * BERT uses learnable positional embeddings

### Pretraining Tasks

* masked language modeling and next sentence prediction

#### Masked Language Modeling

* masking tokens and using tokens from bidirectional context to predict the masked tokens in a self-supervised fashion----> masked language model

* artificial token <mask> will never appear in fine-tuning. 

* To avoid such a mismatch between pretraining and fine-tuning, if a token is masked, in the input it will be replaced with: 
  
  * special <mask> token for 80% of the time
  
  * random token for 10%
  
  * unchanged label token for 10%

* occasional noise encourages BERT to be less biased towards the masked token. 

#### Next Sentence Prediction

* binary classification task to help understand the relationship between 2 text sequaneces--> next sentence prediction



### All together

* BERTEncoder

* MaskLM

* NextSentencePred


