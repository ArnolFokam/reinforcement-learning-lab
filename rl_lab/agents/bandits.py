import numpy as np

from rl_lab.helpers import stable_softmax


class GreedyBandit:
    """
    Greedy bandit that can operate on various environments

    Attributes
    ----------
    reward : int float
        the current average reward

    epsilon : str
        epsilon parameter for e-greedy methods

    timesteps : int
        number of timespteps for exploration

    step_value : int
        step value for the temporal difference learning

    environment : LineWalkEnvironment
        environment to use

    allowed_actions: List[int]
        the different step values to use

    actions_counts: List[int]
        the number times each jumps has been performed

    rewards_per_timesteps: List[float]
        the average reward for the selected action (per timesteps)

    expected_rewards: List[int]
        initial expections of how the reward distribution


    Methods
    -------
    select_action()
        Select the next action to be performed

    jump(step)
        Perform action and update reward

    explore()
        perform exploration over timesteps
    """

    def __init__(
        self,
        allowed_actions,
        timesteps,
        environment,
        epsilon,
        step_value=None,
        **kwargs
    ) -> None:
        """
        Initialize the agent

        Parameters
        ----------

        allowed_actions: List[int]
            the different step values to use

        timesteps : int
            number timespteps for exploration

        environment : LineWalkEnvironment
            environment to use

        epsilon : str, optional
            epsilon parameter for e-greedy methods

        step_value : int, optional
            step value for the temporal difference learning

        has_constant_step:  bool, optional
            does the step value changes over time

        action_selection : str, optional
            the selection process of reward. ["base", "UCB"]

        """

        assert 0 <= epsilon <= 1.0
        assert isinstance(allowed_actions, list) and len(allowed_actions) > 1

        self.epsilon = epsilon
        self.average_reward = 0
        self.timesteps = timesteps
        self.step_value = step_value
        self.environment = environment
        self.allowed_actions = allowed_actions
        self.actions_counts = np.zeros(len(allowed_actions))
        self.rewards_per_timesteps = np.zeros(self.timesteps)
        self.expected_rewards = [
            np.random.random() for _ in range(len(self.allowed_actions))
        ]

    def on_reward_obtained(self, reward, action_idx, step_value):
        """
        Just in case you might to add some
        stuffs after the reward was obtained.
        """
        pass

    def get_step_value(self, action_idx):
        if self.step_value:
            return self.step_value
        else:
            return 1.0 / self.actions_counts[action_idx]

    def select_action(self, step):
        """
        Select the jump to perform

        Parameters
        ----------

        step : int
            current timesptep of the exploration

        Returns
        -------
            int
            index of the selected action
        """
        return np.argmax(self.expected_rewards)

    def act(self, step):
        """Jump and update reward

        Parameters
        ----------

        step : int
            current timesptep of the exploration

        Returns
        -------
            float
            reward obtain at time step 'step'
        """
        if np.random.random() > self.epsilon:
            action_idx = self.select_action(step)
        else:
            action_idx = np.random.randint(len(self.allowed_actions))

        reward = self.environment.get_reward(self.allowed_actions[action_idx])

        self.actions_counts[action_idx] += 1
        step_value = self.get_step_value(action_idx)

        # callbacks for update shenanigans
        # after the calculation of the reward
        self.on_reward_obtained(reward, action_idx, step_value)

        # update the cumulative reward of our agent
        # by recursively computing the average
        self.expected_rewards[action_idx] += (step_value) * (
            reward - self.expected_rewards[action_idx]
        )
        return reward

    def explore(self):
        """
        Explore the environment over timesteps
        """

        for step in range(1, self.timesteps + 1):
            reward = self.act(step)
            self.average_reward = self.average_reward + (1.0 / step) * (
                reward - self.average_reward
            )
            self.rewards_per_timesteps[step - 1] = self.average_reward


class UCBBandit(GreedyBandit):
    """
    Greedy bandit that can operate on various environments

    Attributes
    ----------

    c: float
        parameter that controls the amout of uncertainty that affects the agent
    """

    def __init__(
        self,
        allowed_actions,
        timesteps,
        environment,
        epsilon=0,
        step_value=0.5,
        c=0.1,
        **kwargs
    ) -> None:

        assert c > 0

        super().__init__(
            allowed_actions, timesteps, environment, epsilon, step_value, **kwargs
        )

        self.c = c

    def select_action(self, step):
        """Select the jump to perform

        Parameters
        ----------

        step : int
            current timesptep of the exploration

        Returns
        -------
            int
            index of the selected action
        """
        uncertainty = self.c * np.sqrt(np.log(step) / (self.actions_counts + 1e-19))
        return np.argmax(self.expected_rewards + uncertainty)


class GradientsBandit(GreedyBandit):
    """
    Gradients bandit that can operate on various environments

    """

    def __init__(
        self,
        allowed_actions,
        timesteps,
        environment,
        epsilon=0,
        step_value=0.5,
        **kwargs
    ) -> None:

        super().__init__(
            allowed_actions, timesteps, environment, epsilon, step_value, **kwargs
        )

        self.preferences = np.zeros(len(allowed_actions))

    def on_reward_obtained(self, reward, action_idx, step_value):
        super().on_reward_obtained(reward, action_idx, step_value)
        self.update_preferences(reward, action_idx, step_value)

    def update_preferences(self, reward, action_idx, step_value):
        """
        update prefernecs for the action selection
        """

        # we choose the reward baseline to be the average reward
        reward_baseline = self.average_reward
        probs = stable_softmax(self.preferences)

        # update preference for selected action
        self.preferences[action_idx] += (
            step_value * (reward - reward_baseline) * (1 - probs[action_idx])
        )

        # update preference for other actions
        other_actions_idx = np.delete(range(len(self.allowed_actions)), action_idx)
        self.preferences[other_actions_idx] -= (
            step_value * (reward - reward_baseline) * probs[other_actions_idx]
        )

    def select_action(self, step):
        """Select the jump to perform

        Parameters
        ----------

        step : int
            current timesptep of the exploration

        Returns
        -------
            int
            index of the selected action
        """
        return np.random.choice(
            len(self.allowed_actions),
            1,
            replace=False,
            p=stable_softmax(self.preferences),
        )[0]
