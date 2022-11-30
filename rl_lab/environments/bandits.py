import numpy as np

class LineWalkEnvironment:
    def __init__(self, max_scale, **kwargs) -> None:
        self.max_scale = max_scale
        self.goal = np.random.randint(0, self.max_scale)
        self.agent_position = self.get_new_agent_position(self.max_scale, self.goal)
        
    def get_new_agent_position(self, max_scale, goal):
        # sample the position of the agent while 
        # excluding the goal position
        tmp = list(range(max_scale))
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