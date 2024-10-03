"""
Uses a random agent to collect data.
"""
from abc import ABC, abstractmethod

import gymnasium
from gymnasium import make_vec
import numpy as np
from tqdm import tqdm


class Policy(ABC):
    """
    Temporary base class for policies to implement.
    """
    @abstractmethod
    def act(self, obs):
        """
        Act function takes in an observation and returns an action.
        """
        raise NotImplementedError("Must implement the act method")


class RandomPolicy(Policy):
    """
    Randomly samples from the action space.
    """
    def __init__(self, env: gymnasium.Env):
        self.action_space = env.action_space

    def act(self, _):
        return self.action_space.sample().astype("float64")


def collect_data(n_rollouts: int, n_steps: int, env: gymnasium.Env, policy: Policy):
    """
    Collects data from the env using a policy.
    Performs n rollouts of n steps each.
    """
    data = []
    for _ in tqdm(range(n_rollouts), desc="Collecting rollouts"):
        seeds = np.random.randint(0, 2**32-1, env.num_envs)
        seeds_int = [int(seed) for seed in seeds]
        obses, infos = env.reset(seed=seeds_int)
        for _ in range(n_steps):
            actions = policy.act(obses)
            next_obses, rewards, dones, truncateds, infos = env.step(actions)
            data.append((obses, actions, rewards, next_obses, dones))
            obses = next_obses
    return data


def process_img_data(data: list, save_path: str):
    """
    Obses is a list of numpy arrays of shape (n_steps, n_envs, *img_shape)
    """
    obses, _, _, _, _ = zip(*data)

    obses = np.array(obses)
    obses = obses.reshape(-1, *obses.shape[2:])
    np.save(save_path, obses)


def main():
    """
    Collects n total samples by performing n_rollouts of n_steps each in parallel environments.
    """
    n = 100000
    n_steps = 500
    num_envs = 8
    n_rollouts = n // n_steps // num_envs
    print(f"Collecting {n} samples in {num_envs} x {n_rollouts} rollouts of {n_steps} steps each")
    env = make_vec("CarRacing-v2", num_envs=num_envs)
    policy = RandomPolicy(env)
    data = collect_data(n_rollouts, n_steps, env, policy)
    process_img_data(data, "data/CarRacing-v2/data.npy")


if __name__ == "__main__":
    main()