import numpy as np

class Jumper:
    """
    This is an agent that has the ability to interact in the environment :py:class:`rl_lab.environments.bandits.LineWalkEnvironment`. 
    The agent is can jump to position in that environment given a step value that indicates the amount
    of steps from the current position for the jump. Note that this value can be -/+ ve
    
    Attributes
    ----------
    reward : int float
        the current average reward 
        
    action_selection : str
        the selection process of reward. ["base", "UCB"]
        
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
        the average reward for the selected action over timesteps
        
    expected_rewards: List[int]
        initial expections of how the reward distribution over available actions looks like
    
    
    Methods
    -------
    select_action(step, action_count=0.0)
        Select the next action to be performed
        
    jump(step)
        Perform action and update reward
        
    explore()
        perform exploration over timesteps
    """    
    def __init__(self, 
                 allowed_jumps, 
                 timesteps, 
                 environment, 
                 epsilon = 0, 
                 has_constant_step = False, 
                 step_value = 0.5, 
                 action_selection = "base", # ["base", "UCB"]
                 **kwargs) -> None:
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
        
        self.reward = 0
        self.action_selection = action_selection
        self.epsilon = epsilon
        self.timesteps = timesteps
        self.step_value = step_value
        self.environment = environment
        self.allowed_jumps = allowed_jumps
        self.has_constant_step = has_constant_step
        
        if not self.has_constant_step:
            self.actions_counts = [0] * len(allowed_jumps)
            
        self.rewards_per_timesteps = np.zeros(self.timesteps)
        assert isinstance(allowed_jumps, list) and len(allowed_jumps) > 1
        self.expected_rewards = [np.random.random() for _ in range(len(self.allowed_jumps))]
        
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
        if self.action_selection == "UCB":
            return 0
        else:
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
            step_idx = self.select_action(step)
        else:
            step_idx = np.random.randint(len(self.allowed_jumps))
            
        reward = self.environment.jump(self.allowed_jumps[step_idx])
        
        # update the cumulative reward of our agent
        # by recursively computing the average
        if not self.has_constant_step:
            self.actions_counts[step_idx] += 1  
            self.expected_rewards[step_idx] = self.expected_rewards[step_idx] + (1.0 / self.actions_counts[step_idx]) * (reward - self.expected_rewards[step_idx])
        else:
            self.expected_rewards[step_idx] = self.expected_rewards[step_idx] + self.step_value * (reward - self.expected_rewards[step_idx])
            
        return reward
    
    def explore(self):
        """
        Explore the environment over timesteps
        """
        
        for step in range(1, self.timesteps + 1):
            reward = self.jump(step)
            self.reward = self.reward + (1.0 / step) * (reward - self.reward)
            self.rewards_per_timesteps[step - 1] = self.reward