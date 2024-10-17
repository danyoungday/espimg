
from pathlib import Path

import flappy_bird_gymnasium
import gymnasium
import numpy as np
import torch

from evolution.candidate import Candidate
from evolution.evaluation.evaluator import Evaluator

class RealEvaluator(Evaluator):
    def __init__(self, n_steps=500, n_envs=8, n_repeats=32):
        assert n_repeats % n_envs == 0, "n_repeats must be divisible by n_envs"
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_repeats = n_repeats
        self.env = gymnasium.make_vec("FlappyBird-v0", num_envs=self.n_envs, use_lidar=False)

    def evaluate_candidate(self, candidate: Candidate):
        """
        all_rewards is (n_repeats // n_envs, n_envs) and we just want the mean of all of these
        """
        all_rewards = []
        for _ in range(self.n_repeats // self.n_envs):
            seeds = [int(seed) for seed in np.random.randint(0, 2**32-1, self.n_envs)]
            obses, infos = self.env.reset(seed=seeds)

            dones = np.array([False] * self.n_envs)
            rollout_rewards = np.zeros(self.n_envs)
            i = 0
            while not np.all(dones) and i < self.n_steps:
                # obs_tensor: (n_envs, D)
                obs_tensor = torch.tensor(obses, dtype=torch.float32)
                obs_tensor = obs_tensor.to("mps")
                # actions: (n_envs, 1)
                actions = candidate.prescribe(obs_tensor)
                actions = actions.cpu().numpy().astype("int")
                obses, rewards, terminateds, truncateds, infos = self.env.step(actions)
                dones = np.logical_or(dones, terminateds)
                # Add to rollout rewards if the given env is not done
                rollout_rewards += rewards * (1 - dones)
                i += 1

            all_rewards.append(rollout_rewards)

        all_rewards = np.array(all_rewards)
        total_reward = np.mean(all_rewards)
        candidate.metrics["reward"] = total_reward

        return total_reward
    
    def visualize_evaluate_candidate(self, candidate: Candidate):
        env = gymnasium.make("FlappyBird-v0", use_lidar=False, render_mode="human")
        obs, info = env.reset()
        total_reward = 0
        i = 0
        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = candidate.prescribe(obs_tensor).item()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            i += 1
            if done:
                break
        return total_reward
    
if __name__ == "__main__":

    import pandas as pd

    evaluator = RealEvaluator()
    trial_name = "outerloop"
    n_gens = 7
    results_dir = Path(f"results/{trial_name}/{n_gens}.csv")
    results_df = pd.read_csv(results_dir)
    cand_ids = results_df["cand_id"].values[:10]
    for cand_id in cand_ids:
        cand = Candidate.from_seed(Path(f"results/{trial_name}/{cand_id.split('_')[0]}/{cand_id}.pt"), None, None)
        cand.model.to("cpu")
        reward = 0
        for _ in range(3):
            reward += evaluator.visualize_evaluate_candidate(cand)
        reward /= 3
        print(cand_id, reward)