import numpy as np


def run_bandits_experiments(agents_specs, environment, timesteps, num_actions, num_experiments, max_line_length):
        
    agents = {
        agent: {
            "params": specs["params"], #  parameters of the learning agent
            "color": specs["color"], # color for visualization
            "average_reward_per_timesteps": np.zeros(timesteps),
            "average_reward_per_actions": np.zeros(num_actions),
            } for agent, specs in agents_specs.items()
        }
    
    for idx in range(1, num_experiments + 1):
    
        env = environment(max_line_length)

        # create available 10 jumps such that 
        # there exists only few possible jumps 
        actions = np.random.randint(-max_line_length, max_line_length, num_actions)
        
        # sort action in increasing reward 
        rewards_per_actions = np.vectorize(lambda action: env.get_reward(action))(actions)
        sorted_actions_index = np.argsort(rewards_per_actions)
        actions = actions[sorted_actions_index]
        
        for name in agents:
            # create a new agent for each experiment
            # all agent interact on the same environment (but not at once).
            agent = agents_specs[name]["agent"](list(actions), timesteps, env, **agents[name]["params"])
            agent.explore()
            
            # update metrics average over experiments
            agents[name]["average_reward_per_timesteps"] = agents[name]["average_reward_per_timesteps"] + (1.0 / idx) * (agent.rewards_per_timesteps - agents[name]["average_reward_per_timesteps"])
            agents[name]["average_reward_per_actions"] = agents[name]["average_reward_per_actions"] + (1.0 / idx) * (agent.expected_rewards - agents[name]["average_reward_per_actions"])
            
    return agents
        