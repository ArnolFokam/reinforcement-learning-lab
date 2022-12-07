- In every reinfrocement learning system, an agent interact at series of iscrete time steps.
- At each time step *t*, a *feedback* ***(State)*** is received from the environment which causes an update in selection the agent's actions.
- In other words, the agent learns to map states to probailities of selecting eacj possible action. This mapping is the agent's ***policy****.
- Also, the timesteps need not to be fixed. It can also be successive stages of a decision making process. **Example: Show the next best task to perform after completing one.**
- The Agent-Environment relationship is abstract and should be confused with nature's interpretation of an environment. **Example *Robot walking* algorithm could have as model  the decision making policy but its environment could include its joints as well as obstacles even the we tend to interpret the joints as part of the robot (model)**.

## Rewards
- Rewards should tell the goal to be acheived but not bias an agent towards a particular way to acheive this goal. **Example: An agent should be given rewards when he wins or not. Adding rewards for things such as controlling the center of the board or material can bias the agent towards a particular playstyle which might be detrimental in the long run.**
- RL tasks can be seperated into *episodic* and *continuing* tasks. *Episodic* tasks are tasks that can be represented though fixed sequential timesteps which terminal and non-terminal states while *Continuing* contains no such terminal boundaries.
```
| Episodic                 | Contuining  |
| ------------------------ | ----------- |
| Playing board games      |   Home robot       |
| Solving a puzzle         |   Self-Driving     |
```

- There for continuining tasks, we would like to model the experted reward return. However we can use the formula below which is the sum of the expected future rewards.
$$
G_t=R_{t+1}+R_{t+2}+R_{t+3}+\cdots+R_T
$$
- This is because of non-terminal nature of these tasks which will result in $G_t=\infty$
- A more appropriate function includes the a discount rate $\gamma$  which regulate the effect expected future rewards have on the current timestep *t*.
$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots=\sum_{k=0}^{\infty} \gamma^k R_{t+k+1},
$$
where $\gamma$ is a parameter, $0 \leq \gamma \leq 1$, called the discount rate.
- As $\gamma$ tends towards $0$, we give more priorities to the next epected reward. But as $\gamma$ tends towards 1, we foresight farther into the future.

## The Markov Property
- An important property of RL framework is the agent's ability to get feedbacks from environment.
- This ability is not defined by the quantity of information to be received (even though more information is always better), but the agent's ability to retain useful information over time to accomplish its goal.
- If we assume the state signal has a the *Markov property*, then, the signal obtained at $t + 1$ depends on the the signals obtianed at $t$ only. There we can model our reward function according to:
$$
p(s^{\prime}, r \mid s, a) = \operatorname{Pr}\left\{R_{t+1}=r, S_{t+1}=s^{\prime} \mid R_t, S_t, A_t\right\}
$$
- Another way to formulate this is that, if a state signal obeys the Markov property, then we only need the current stat to predict the next state.
- If we had the best poslicy, the effieciency of using the current state ony would be as good as if we used the previous state as well.
- It is important to not that we can still use approximation of Markov models when the state signal is not one.
