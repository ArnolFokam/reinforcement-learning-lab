- While other learning methods *give* the correct action(s) during learning, reinforcement aims at *evaluating* the actions taken.
- Methods like supervised learning gives the correction action to be taken regardless of the correction action taken while reinforcement learning aims at evaluation the action.
- In reinforcement learning, we do what was described in the previous point because we do not know what actions to take to acheive our goal.

## n-Armed Bandit Problem
- This is  a learning problem where we can perfrom **n actions** with the aim of maximizing an expected gain over a period of time and **each action has an expected gain (if selected)**  that must be learned by our agent through exploration.
- One way to solve this problem is to greedily choose the action that has the maximum expected gain (greedy). 
- However, a problem to this approach arises at the begining of the process when we have no initial knowledge of how any actions influences our gains. Also, selecting the best action might not necessarily lead to the most optimal solution in a long run.
- Another popular method is to continue being greedy while ocasionally sampling other actions from our set with a probability $\epsilon$.
- 