from typing import Any, Dict

import numpy as np

from rl_lab.environments.bandits import Environment


def run_bandits_experiments(
    agents_specs: Dict[str, Any],
    environment: Environment,
    timesteps,
    num_actions,
    num_experiments,
    line_length,
) -> Dict[str, Any]:
    """
    Run multiple bandit experiments with
    different agents but the same environment.

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
        results obtained from the experiments.
    """

    agents = {
        agent: {
            "params": specs["params"],  # parameters of the learning agent
            "color": specs["color"],  # color for visualization
            "timesteps_reward": np.zeros(timesteps),
            "actions_reward": np.zeros(num_actions),
        }
        for agent, specs in agents_specs.items()
    }

    for idx in range(1, num_experiments + 1):

        env = environment(line_length)

        # create available 10 jumps such that
        # there exists only few possible jumps
        actions = np.random.randint(-line_length, line_length, num_actions)

        # sort action in increasing reward
        rewards_per_actions = np.vectorize(env.get_reward)(actions)
        sorted_actions_index = np.argsort(rewards_per_actions)
        actions = actions[sorted_actions_index]

        for name in agents:
            # create a new agent for each experiment
            # all agent interact on the same environment (but not at once).
            agent = agents_specs[name]["agent"](
                list(actions), timesteps, env, **agents[name]["params"]
            )
            agent.explore()

            # update rewards average (per timesteps) over experiments
            agents[name]["timesteps_reward"] += +(1.0 / idx) * (
                agent.rewards_per_timesteps - agents[name]["timesteps_reward"]
            )  # noqa

            # update rewards average (per actions) over experiments
            agents[name]["actions_areward"] += (1.0 / idx) * (
                agent.expected_rewards - agents[name]["actions_reward"]
            )  # noqa

    return agents
