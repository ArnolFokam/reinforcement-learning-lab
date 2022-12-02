from typing import Any, Dict
import numpy as np

from rl_lab.environments.bandits import Environment

def run_bandits_experiments(agents_specs: Dict[str, Any], 
                            environment: Environment, 
                            timesteps, num_actions, 
                            num_experiments, line_length) -> Dict[str, Any]:
    """
    Run multiple bandit experiments with different agents but the same environment.

    Parameters
    ----------
        
    agents_specs : AgentSpecs 
        specification of various agents that will interact with environment. 
        This is a dictionay of agent configurations.
        
        Example:

        specs = {"agent name": {
            "agent": AgentClass,
            "params": {
                agent class parameters...
            },
            "color": "red"
        }}

        
    environment: Environment
        environment in which the agents will evolve.
        
    timesteps : int
        number of timespteps for exploration
        
    num_actions : int
        number of actions that the agents can perform
        
    num_experiments : int
        number of experiments to perform for each agent
        
    line_length : int
        length of the line on which our agents are positioned
    
    Returns
    -------
    Dict[str, Any]
        results of the data. The results are just dictionary of list of values. 
        Where each key is an agent and each list contains a specific set of values 
        represent the results of our experiments. Additional, this data structure 
        contains any other information required for plotting. It mostly consists 
        of properties copied from the 'specs' data structure
    """
    
    agents = {
        agent: {
            "params": specs["params"], #  parameters of the learning agent
            "color": specs["color"], # color for visualization
            "average_reward_per_timesteps": np.zeros(timesteps),
            "average_reward_per_actions": np.zeros(num_actions),
            } for agent, specs in agents_specs.items()
        }
    
    for idx in range(1, num_experiments + 1):
    
        env = environment(line_length)

        # create available 10 jumps such that 
        # there exists only few possible jumps 
        actions = np.random.randint(-line_length, line_length, num_actions)
        
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
        