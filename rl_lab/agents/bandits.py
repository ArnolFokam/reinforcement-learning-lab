import numpy as np


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

    allowed_jumps: List[int]
        the different step values to use

    has_constant_step:
        does the step value changes over time

    actions_counts: List[int]
        the number times each jumps has been performed

    rewards_per_timesteps: List[float]
        the average reward for the selected action (per timesteps)

    expected_rewards: List[int]
        initial expections of how the reward distribution


    Methods
    -------
    select_action(step, action_count=0.0)
        Select the next action to be performed

    jump(step)
        Perform action and update reward

    explore()
        perform exploration over timesteps
    """

    def __init__(
        self,
        allowed_jumps,
        timesteps,
        environment,
        epsilon=0,
        has_constant_step=False,
        step_value=0.5,
        **kwargs
    ) -> None:
        """
        Initialize the agent

        Parameters
        ----------

        allowed_jumps: List[int]
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
        assert isinstance(allowed_jumps, list) and len(allowed_jumps) > 1

        self.reward = 0
        self.epsilon = epsilon
        self.timesteps = timesteps
        self.step_value = step_value
        self.environment = environment
        self.allowed_jumps = allowed_jumps
        self.has_constant_step = has_constant_step
        self.actions_counts = [0] * len(allowed_jumps)
        self.rewards_per_timesteps = np.zeros(self.timesteps)
        self.expected_rewards = [
            np.random.random() for _ in range(len(self.allowed_jumps))
        ]

    def get_step_value(self, action_idx):
        if self.has_constant_step and self.step_value:
            return self.step_value
        else:
            return 1.0 / self.actions_counts[action_idx]

    def select_action(self):
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
        return np.argmax(self.expected_rewards)

    def jump(self, step):
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
            action_idx = np.random.randint(len(self.allowed_jumps))

        reward = self.environment.get_reward(self.allowed_jumps[action_idx])

        # update the cumulative reward of our agent
        # by recursively computing the average
        self.actions_counts[action_idx] += 1
        step_value = self.get_step_value(action_idx)
        self.expected_rewards[action_idx] += (step_value) * (
            reward - self.expected_rewards[action_idx]
        )  # noqa
        return reward

    def explore(self):
        """
        Explore the environment over timesteps
        """

        for step in range(1, self.timesteps + 1):
            reward = self.jump(step)
            self.reward = self.reward + (1.0 / step) * (reward - self.reward)
            self.rewards_per_timesteps[step - 1] = self.reward


class UCBBandits(GreedyBandit):
    """
    Greedy bandit that can operate on various environments

    Attributes
    ----------

    confidence_level: float
        parameter that controls the amout of uncertainty that affects the agent
    """

    def __init__(
        self,
        allowed_jumps,
        timesteps,
        environment,
        epsilon=0,
        has_constant_step=False,
        step_value=0.5,
        confidence_level=0.1,
        *kwargs
    ) -> None:

        assert confidence_level > 0

        super().__init__(
            allowed_jumps,
            timesteps,
            environment,
            epsilon,
            has_constant_step,
            step_value,
            **kwargs
        )

        self.confidence_level = self.confidence_level

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
        uncertainty = self.confidence_level * np.sqrt(
            np.log(step) / self.actions_counts
        )
        return np.argmax(self.expected_rewards + uncertainty)
