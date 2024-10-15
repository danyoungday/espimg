"""
Uses a random agent to collect data.
"""
from abc import ABC, abstractmethod

import gymnasium
from gymnasium import make_vec
from gymnasium.utils.play import play
import numpy as np
from tqdm import tqdm


# from evolution.candidate import Candidate


class Rollout:
    """
    Dataclass to store one rollout of data.
    Takes in lists of states from the environment
    obses: (L, H, W, C)
    actions: (L, A)
    rewards: (L, 1)
    next_obses: (L, H, W, C)
    """
    def __init__(self, obses, actions, rewards, next_obses):
        self.obses = np.array(obses)
        self.actions = np.array(actions)
        if len(self.actions.shape) == 1:
            self.actions = self.actions.reshape(-1, 1)
        self.rewards = np.array(rewards).reshape(-1, 1)
        self.next_obses = np.array(next_obses)

        assert self.obses.shape[0] == self.actions.shape[0] == self.rewards.shape[0] == self.next_obses.shape[0]

    @classmethod
    def from_padded(cls, obses, actions, rewards, next_obses):
        """
        To load from padded numpy arrays construct a dummy rollout then replace its attributes. Then unpad them.
        """
        rollout = cls([0], [0], [0], [0])
        rollout.obses = obses
        rollout.actions = actions
        rollout.rewards = rewards
        rollout.next_obses = next_obses
        rollout.unpad()
        return rollout

    def pad_arr(self, arr, pad_length):
        padded = arr
        if arr.shape[0] < pad_length:
            pad_amt = pad_length - arr.shape[0]
            pad_width = ((0, pad_amt), *[(0, 0) for _ in range(arr.ndim - 1)])
            padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
    
        return padded

    def pad(self, pad_length: int):
        """
        Pads the rollout to a specified length.
        """
        self.obses = self.pad_arr(self.obses, pad_length)
        self.actions = self.pad_arr(self.actions, pad_length)
        self.rewards = self.pad_arr(self.rewards, pad_length)
        self.next_obses = self.pad_arr(self.next_obses, pad_length)

    def unpad_arr(self, arr):
        """
        Unpads a single numpy array
        From ChatGPT
        """
        axes_to_check = tuple(range(1, arr.ndim))
    
        # Check if any non-zero elements exist along the specified axes
        non_padding = np.any(arr != 0, axis=axes_to_check)
        
        # Find the first index where the 0th dimension slice is all zero
        original_length = np.argmax(non_padding == False)
        
        # If no padding is found (no zero rows), return the full length
        if non_padding.all():
            return arr
        
        # Slice the array to the original length along the 0th dimension
        arr_unpadded = arr[:original_length]
        return arr_unpadded
    
    def unpad(self):
        """
        Unpads the rollout.
        """
        self.obses = self.unpad_arr(self.obses)
        self.actions = self.unpad_arr(self.actions)
        self.rewards = self.unpad_arr(self.rewards)
        self.next_obses = self.unpad_arr(self.next_obses)

    def __len__(self):
        return self.obses.shape[0]


def save_rollouts(rollouts: list[Rollout], path):
    """
    Saves rollouts to a compressed numpy file.
    """
    longest = max([len(rollout) for rollout in rollouts])
    for rollout in rollouts:
        rollout.pad(longest)
    
    all_obses = np.stack([rollout.obses for rollout in rollouts], axis=0)
    all_actions = np.stack([rollout.actions for rollout in rollouts], axis=0)
    all_rewards = np.stack([rollout.rewards for rollout in rollouts], axis=0)
    all_next_obses = np.stack([rollout.next_obses for rollout in rollouts], axis=0)

    np.savez_compressed(path, obses=all_obses, actions=all_actions, rewards=all_rewards, next_obses=all_next_obses)


def load_rollouts(path):
    """
    Loads rollouts back to list from compressed file.
    """
    data = np.load(path)
    obses = data["obses"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_obses = data["next_obses"]
    rollouts = []
    for obs, action, reward, next_obs in zip(obses, actions, rewards, next_obses):
        rollouts.append(Rollout.from_padded(obs, action, reward, next_obs))

    return rollouts


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
        return self.action_space.sample().astype("int")

# class CandidatePolicy(Policy):
#     """
#     Selects action based on the candidate prescriptor.
#     """
#     def __init__(self, candidate: Candidate):
#         self.candidate = candidate

#     def act(self, obs):
#         return self.candidate.prescribe(obs)


def collect_data(n_rollouts: int, n_steps: int, env: gymnasium.Env, policy: Policy, num_envs=1, continuous=False) -> list[np.ndarray]:
    """
    Collects data from the env using a policy.
    Performs n rollouts of n steps each.
    """
    data = [[], [], [], []]
    for _ in tqdm(range(n_rollouts), desc="Collecting rollouts"):
        rollouts = [[], [], [], []]
        seeds = np.random.randint(0, 2**32-1, num_envs)
        seeds_int = [int(seed) for seed in seeds]
        obses, infos = env.reset(seed=seeds_int)
        for _ in range(n_steps):
            actions = policy.act(obses)
            next_obs_obses, rewards, dones, truncateds, infos = env.step(actions)

            rollouts[0].append(obses)
            if continuous:
                rollouts[1].append(actions)
            else:
                rollouts[1].append(np.array(actions).reshape(-1, 1))
            rollouts[2].append(np.array(rewards).reshape(-1, 1))
            rollouts[3].append(next_obs_obses)

            obses = next_obs_obses

        # (4, n_steps, num_envs, D)
        rollouts = [np.stack(rollout, axis=0) for rollout in rollouts]

        data[0].append(rollouts[0])
        data[1].append(rollouts[1])
        data[2].append(rollouts[2])
        data[3].append(rollouts[3])

    # (4, n_rollouts, n_steps, num_envs, D)
    data = [np.stack(rollouts, axis=0) for rollouts in data]
    # (4, n_rollouts, num_envs, n_steps, D)
    data = [rollouts.transpose(0, 2, 1, *list(range(3, len(rollouts.shape)))) for rollouts in data]
    # (4, n_rollouts * num_envs, n_steps, D) = (4, N, n_steps, D)
    data = [rollouts.reshape(-1, *rollouts.shape[2:]) for rollouts in data]

    return data


def record_states(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    """
    Function called at every step of the process.
    * obs_t: observation before performing action
    * obs_tp1: observation after performing action
    * action: action that was executed
    * rew: reward that was received
    * terminated: whether the environment is terminated or not
    * truncated: whether the environment is truncated or not
    * info: debug info
    """
    human_data.append([obs_t, action, rew, obs_tp1, terminated, truncated])

def discrete_to_continuous(action: int) -> np.ndarray:
    if action == 0:
        return np.array([0.0, 0.0, 0.0])
    elif action == 1:
        return np.array([1.0, 0.0, 0.0])
    elif action == 2:
        return np.array([-1.0, 0.0, 0.0])
    elif action == 3:
        return np.array([0.0, 1.0, 0.0])
    elif action == 4:
        return np.array([0.0, 0.0, 1])
    raise ValueError("Invalid action")

def collect_human_data(env: gymnasium.Env, keys_to_action: dict, continuous=False):
    global human_data
    human_data = []  # List size (num_rollouts * num_steps, 6)

    play(env, keys_to_action=keys_to_action, callback=record_states)

    rollouts = []
    obses, actions, rewards, next_obses = [], [], [], []
    for obs, action, reward, next_obs, terminated, truncated in human_data:
        if isinstance(obs, tuple) or isinstance(obs, list):  # Super weird bug with data collection
            obs = obs[0]
        obses.append(obs)
        if continuous:
            action = discrete_to_continuous(action)
        actions.append(action)
        rewards.append(reward)
        next_obses.append(next_obs)

        if terminated or truncated:
            rollout = Rollout(obses, actions, rewards, next_obses)
            rollouts.append(rollout)
            obses, actions, rewards, next_obses = [], [], [], []

    if len(obses) > 0:
        rollout = Rollout(obses, actions, rewards, next_obses)
        rollouts.append(rollout)

    print(rollouts[0].obses.shape, rollouts[0].actions.shape, rollouts[0].rewards.shape, rollouts[0].next_obses.shape)
    print(rollouts[1].obses.shape, rollouts[1].actions.shape, rollouts[1].rewards.shape, rollouts[1].next_obses.shape)

    return rollouts


def main():
    env = gymnasium.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
    collect_human_data(env, keys_to_action={"w": 3, "a": 2, "d": 1, "s": 4}, continuous=True)


if __name__ == "__main__":
    main()