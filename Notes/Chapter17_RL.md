# Chapter 17 RL

## 17.1 Markov Decision Process (MDP)

### MDP

* A model for how the state of a system evolves as different actions are applied to the system.

### Return and Discount Factor

* MDP: $(S, A, T, r)$. 
  
  * trajectory: $\tau = (s_0, a_0, r_0, ...)$
  
  * reward: $r_t = r(s_t, a_t)$
  
  * Total reward: $R(\tau) = r_0 + r_1 + ...$. 

* RL: find a trajectory with largest return (total reward)

* discounted return: $R(\tau) = r_0, \gamma r_1 + \gamma^2 r_2 +... = \sum_{t=0}^{\infty} \gamma^t r_t$

* 
