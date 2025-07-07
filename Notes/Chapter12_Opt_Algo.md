# Chapter 12: Optimization Algorithms



## 12.1 Optimization and Deep Learning

* Loss function definition firstly
  
  * It is also the objective function of the optimization problem

* Most are minimization problem

* Want a maximization? flip the sign on the objective

### Goal of Optimization

* Reduce the generalization error
  
  * also pay attention to overfitting

* Empirical risk and the risk
  
  * empirical risk: loass on training set:  g
  
  * risk: entire population of data:  f



### Optimization Challenges in DL

* performance of optimization algorithms in minimizing the objective function

* use numerical optimization algo here (in real world, most objective functions are complicated and do not have analytical solutions)

#### Local Minima

* When the solution is near the local optimum, it may only minimize the objective function locally rather than globally

* Only some degree of noise might knock the parameter out of the local minimum

* Minibatch stochastic gradient descent where the natrual variation of gr4adients over minibatches is able to dislodge the parameters from local minima

#### Saddle Points:

* Another reason for gradient vanish

* Determination of the solution status: hessian matrix
  
  * Local minimum: eigenvalues of the functions' Hessian matrix at the zero-gradient position are all positive
  
  * Local maximum: ... are all negative
  
  * saddle point: ... are negative and positive

#### Vanishing Gradients

* Optimization get stuck for a long time before making progress





## 12.2 Convexity

### Definitions

#### Convex Sets

* for all $\lambda \in [0, 1]$, we have $\lambda a + (1 - \lambda)b \in X \space whenever \space a, b \in X$

#### Convex Functions

* convex function f is convex

* $\lambda f(x) + (1 - \lambda) f (x') \geq f(\lambda x + (1-\lambda )x') $

#### Jensen's Inequality

* A generalization of the definition of convexity: 
  
  * $\sum_i \alpha_if(x_i) \geq f(\sum_i\alpha_ix_i) \space and \space E_X[f(X)] \geq f(E_X[X])$



### Properties

#### Local Minima Are Global Minima

#### Below Sets of Convex Functions Are Convex

#### Convexity and Second Derivatives



### Constraints

* Convex optimization allows us to handle constraints efficiently
  
  * ${minimize \atop x} \sim f(x) \\ subject \space to \space c_i(x) \geq 0 for \space all \space i \in {1, ..., n}$

#### Lagrangian

* adapt the Lagrangain L:  $\alpha_ic_i(x)$

#### Penalities

* add ${\lambda \over 2} {||\bold {w}||^2}$ to the objective function to ensure w does not grow too large.

#### Projections

* $\bold{g} \larr \bold{g} \cdot min(1, \theta /||\bold{g}||)$

* A projection on a convex set X: 
  
  * $Proj_x(x) = {argmin \atop {x' \in \textit{X}}}||x - x'||$




