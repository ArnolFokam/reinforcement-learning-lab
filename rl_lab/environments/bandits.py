import numpy as np


class BernoulliEnvironment:
    def cast(self, reward):
        """
        Cast the reward obtained accoring to a Bernoulli distribution
        
        Parameters
        ----------
        
        step : int
            step value of the agent's jump

        """
        
        if np.random.random() < reward:
            return 1
        else:
            return 0

class LineWalkEnvironment(BernoulliEnvironment):
    """
    This is the environment where the agent :py:class:`rl_lab.agents.bandits.Jumper` evolve. 
    This environment is just a basic armed-bandit task where we an agent at a particular position
    on a line and the agent can jump to another location with the objective of reaching the a goal position.
    The agent can only jump one time and the reward is calculated based on difference between the new positon and the goal position.
    
    Attributes
    ----------
        
    line_length : int
        length of the line on which our agent is positioned
        
    goal_position : int
        goal position that agent must reach or be near as much as possible
        
    agent_position : int
        initial position of agent on the line
        
    Methods
    -------
    select_action(step, action_count=0.0)
        Select the next action to be performed
        
    jump(step)
        Perform action and update reward
        
    explore()
        perform exploration over timesteps
    """
    
    def __init__(self, line_length, **kwargs) -> None:
        """
        Initialize the environment by randomly chosing:
            - The length of our line
            - The position of our agent

        Parameters
        ----------
        
        line_length : int
            max length of the line of which our agent is position

        """
        
        self.line_length = line_length
        self.goal_position = np.random.randint(1, self.line_length)
        self.agent_position = self.get_new_agent_position(self.line_length, self.goal_position)
        
    def get_new_agent_position(self, line_length, goal):
        """
        Randomly sample the agent position from a line scale

        Parameters
        ----------
        
        line_length : int
            length of the line on which our agent is positioned

        """
        # sample the position of the agent while 
        # excluding the goal position
        tmp = list(range(line_length))
        tmp.remove(goal)
        return np.random.choice(tmp)
    
    def get_reward(self, step):
        """
        Calculate the true reward from the jump

        Parameters
        ----------
        
        step : int
            step value of the agent's jump

        """
        agent_position = self.agent_position + step
        return 1 / (np.abs(self.goal_position - agent_position) + 1)
        
    def jump(self, step):
        """
        Cast the reward obtained accoring to a Bernoulli distribution
        
        Parameters
        ----------
        
        step : int
            step value of the agent's jump

        """
        reward = self.get_reward(step)
        return self.cast(reward)