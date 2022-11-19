- While other learning methods *give* the correct action(s) during learning, reinforcement aims at *evaluating* the actions taken.
- Methods like supervised learning gives the correction action to be taken regardless of the correction action taken while reinforcement learning aims at evaluation the action.
- In reinforcement learning, we do what was described in the previous point because we do not know what actions to take to acheive our goal.

## n-Armed Bandit Problem
- This is  a learning problem defined by the ability to perform **n actions** $a = 1, 2, \cdots, n$ with the aim of maximizing a total expected gain $A_t$ and **each action has an expected gain (if selected)**  $Q_t(a)$ that must be learned by our agent through exploration.

$$
A_t=\underset{a}{\arg \max } Q_t(a)
$$

- One way to solve this problem is to greedily choose the action that has the maximum expected gain (greedy). 
- However, one problem of this approach is that we have no initial knowledge of how any actions influences our gains. Therefore, we will inevitably always rely on random guesses which might not necessarily lead to the most optimal solution in a long run.
- Another popular method is to continue being greedy while ocasionally sampling other actions from our set with a probability $\epsilon$. Such methods are called $\epsilon$-greedy methods.