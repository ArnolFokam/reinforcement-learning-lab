import numpy as np

class LineWalkEnvironment:
    def __init__(self, max_line_length, **kwargs) -> None:
        self.max_line_length = max_line_length
        self.goal = np.random.randint(0, self.max_line_length)
        self.agent_position = self.get_new_agent_position(self.max_line_length, self.goal)
        
    def get_new_agent_position(self, max_line_length, goal):
        # sample the position of the agent while 
        # excluding the goal position
        tmp = list(range(max_line_length))
        tmp.remove(goal)
        return np.random.choice(tmp)
    
    def get_reward(self, step):
        agent_position = self.agent_position + step
        return 1 / (np.abs(self.goal - agent_position) + 1)
        
    def jump(self, step):
        """We jump according to a bernoulli distribution"""
        reward = self.get_reward(step)
        
        if np.random.random() < reward:
            return 1
        else:
            return 0