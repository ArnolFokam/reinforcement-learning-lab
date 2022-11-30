import numpy as np

class Jumper:
    def __init__(self, 
                 allowed_jumps, 
                 timesteps, 
                 environment, 
                 epsilon = 0, 
                 has_constant_step = False, 
                 step_value = 0.5, 
                 action_selection = "base", # ["base", "UCB"]
                 **kwargs) -> None:
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
        
    def select_next_action(self, step, action_count=0.0):
        if self.action_selection == "UCB":
            return 0
        else:
            return np.argmax(self.expected_rewards)

    def jump(self, step):
        if np.random.random() > self.epsilon:
            step_idx = self.select_next_action(step)
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
        
        for step in range(1, self.timesteps + 1):
            reward = self.jump(step)
            self.reward = self.reward + (1.0 / step) * (reward - self.reward)
            self.rewards_per_timesteps[step - 1] = self.reward