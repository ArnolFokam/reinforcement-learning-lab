- While other learning methods *give* the correct action(s) during learning, reinforcement aims at *evaluating* the actions taken.
- Methods like supervised learning gives the correction action to be taken regardless of the correction action taken while reinforcement learning aims at evaluation the action.
- In reinforcement learning, we do what was described in the previous point because we do not know what actions to take to acheive our goal.

## n-Armed Bandit Problem
- This is  a learning problem defined by the ability to perform **n actions** $a = 1, 2, \cdots, n$ with the aim of maximizing a total expected gain $A_t$ and **each action has an expected gain (if selected)**  $Q_t(a)$ that must be learned by our agent through exploration.

$$
A_t=\underset{a}{\arg \max } Q_t(a)
$$

- We do this by sampling action from our action space and evaluating the reward obtained from using that action. By sampling differenct actions enough times, we obtain a rough estimate of the true reward of each actions.
- A way to calculate this true reward is by averaging the rewards obtained since the begining of the task/game/experiment.
$$

Q_t(a)=\frac{R_1+R_2+\cdots+R_{N_t(a)}}{N_t(a)}

$$
- Keep in mind that $N_t(a)$ is the number times action $a$ has been taken until the $t^{th}$ timestep.
- From an engineering perspective, it would be ineffiecient to save the rewards obtained and compute the average reward  at each timestep. Therefore, instead of using the formula above, we use its recursive brother that only depends on the **current** and **previous** reward as well as the number of times the action has be taken.
$$

Q_t(a)= Q_{t - 1}(a)+ \frac{1}{N_t(a)} (R_{N_t(a)} - Q_{t - 1}(a))

$$
- This in effect says that, to get a new estimate take what we have previous and *a fraction of our error in way thought we should have gotten*.
- If you play close attention to the function, you can see a pattern. Let us say we replace the fraction in the second term by a fraction $\alpha$
$$

Q_t(a)= Q_{t - 1}(a)+ \alpha (R_{N_t(a)} - Q_{t - 1}(a))

$$
- When  $\alpha=0$, we are literally telling our agent to forget about the <span style="color:green"> next reward</span>, all what matters is the <span style="color:blue">previous reward</span>. However, when $\alpha=1$, we are telling it to forget about the <span style="color:blue">previous reward</span>, what matters is the <span style="color:green">next reward</span>.
- Therefore $\alpha$ is parameter the influence the amout influence past rewards have on future rewards during decision making.
- When this parameters is set to $\frac{1}{N_t(a)}$, calculating $Q_t(a)$ gives average reward for that action according to a uniform distribution.
- Setting this parameter to a constant value if beneficial to non-stationary problems where the rewards can change over time because this gives the ability to our agent to focus on the most recent reward obtained from a peraticular action.
### Greed and $\epsilon$-Greedy Methods
- One way to solve this problem is to greedily choose the action that has the maximum expected gain **(greedy)**.
- However, one problem of this approach is that we have no initial knowledge of how any actions influences our gains. Therefore, we will inevitably always rely on random guesses which might not necessarily lead to the most optimal solution in a long run.
- Another popular method is to continue being greedy while ocasionally sampling other actions from our set with a probability $\epsilon$. Such methods are called $\epsilon$-greedy methods.
- Mult-armed bandit problems are good problems that helps us showcase the need for a balance between exploration and exploitation in reinforcement learning
-  $\epsilon$-greedy methods for example are parametrized by $\epsilon$ that influence the degree of curiousity of our agent (ability to explore different actions even though we know a suboptimal action).
