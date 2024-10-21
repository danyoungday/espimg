import flappy_bird_gymnasium
import gymnasium
import numpy as np
import torch

from evolution.candidate import Candidate
from evolution.evaluation.evaluator import Evaluator
from lstm.lstmmodel import LSTM
from flappy.data import Policy, RandomPolicy, CandidatePolicy, collect_data, Rollout
from flappy.train import train_lstm, LSTMDataset
from flappy.env import LSTMVecEnv

class LSTMEvaluator(Evaluator):
    """
    Evaluates candidates in the LSTM world model.
    We draw a sample from the true env and use it to start the LSTM.
    """
    def __init__(self, epochs=100,
                 n_rollouts=1000,
                 n_steps=500,
                 n_envs=32,
                 gamma=0.25,
                 prune=0.9,
                 device="cpu",
                 log_path=None):
        self.epochs = epochs
        self.gamma = gamma
        self.prune = prune
        self.n_rollouts = n_rollouts
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.env = gymnasium.make("FlappyBird-v0", use_lidar=False)
        self.device = device
        self.log_path = log_path

        # All rollouts is a list of each generation of rollouts
        self.all_rollouts = [self.collect_rollouts([RandomPolicy()])]
        self.lstm_env = self.train(self.all_rollouts[0])

    def evaluate_candidate(self, candidate: Candidate):
        """
        obses: (n_envs, 1, D)
        actions: (n_envs, 1, 1)
        rewards: (n_envs, 1, 1)
        terminateds: (n_envs, 1, 1)
        dones: (n_envs)
        total_rewards: (n_envs)
        """
        obses = self.lstm_env.reset()
        dones = torch.zeros(self.n_envs, device=self.device).bool()
        total_rewards = torch.zeros(self.n_envs, device=self.device)
        i = 0
        while not torch.all(dones) and i < self.n_steps:
            actions = candidate.prescribe(obses)
            obses, rewards, terminateds = self.lstm_env.step(actions)
            total_rewards += rewards.squeeze() * (~dones).int()
            dones = torch.logical_or(dones, terminateds.squeeze())
            i += 1

        total_reward = total_rewards.mean().item()
        candidate.metrics["reward"] = total_reward

    def collect_rollouts(self, policies: list[Policy]) -> list[Rollout]:
        assert self.n_rollouts % len(policies) == 0
        rollouts = []
        for policy in policies:
            rollouts.extend(collect_data(policy, self.n_rollouts // len(policies), n_envs=8, n_steps=self.n_steps))

        # Log the rewards of the collected rollouts
        if self.log_path:
            rewards = torch.FloatTensor([torch.mean(rollout.rewards) for rollout in rollouts])
            rewards = rewards.view(len(policies), self.n_rollouts // len(policies))
            cand_rewards = torch.mean(rewards, dim=1)
            cand_rewards = cand_rewards.tolist()
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{cand_rewards}\n")

        return rollouts
    
    def prune_data(self, all_rollouts: list[list[Rollout]], prune: float) -> list[Rollout]:
        """
        Reduces the length of each list of rollout by a factor of prune.
        We just chop off the end which is the same as randomly doing it.
        """
        pruned_rollouts = []
        for rollout_list in all_rollouts:
            pruned_rollouts.append(rollout_list[:int(len(rollout_list) * prune)])
        return pruned_rollouts

    def train(self, rollouts: list[Rollout]):
        """
        Retrains LSTM with rollouts
        """
        dataset = LSTMDataset(rollouts, gamma=self.gamma)
        new_lstm = LSTM(12, 1, 256)
        rew_loss, obs_loss, term_loss = train_lstm(new_lstm, dataset, self.epochs, 64, None, device=self.device)
        print(f"Reward loss: {rew_loss[-1]}, Z loss: {obs_loss[-1]}, Term loss: {term_loss[-1]}")
        new_lstm.to(self.device)
        new_lstm.eval()
        lstm_env = LSTMVecEnv(new_lstm, self.env, self.n_envs, device=self.device)
        return lstm_env
    
    def retrain_lstm(self, candidates: list[Candidate]):
        """
        Retrains lstm on a set of candidates.
        First converts them to a policy and collects data from them.
        Then prunes the old data and appends the new data to the old data.
        Finally retrains the lstm on the total dataset.
        """
        # Convert candidates into policies that collect_data can read
        policies = [CandidatePolicy(c) for c in candidates]
        # Collect a generation of rollouts from the real world
        rollouts = self.collect_rollouts(policies)
        # Prune each old generation of rollouts
        self.all_rollouts = self.prune_data(self.all_rollouts, self.prune)
        # Append the current generation of rollouts to the list of generations of rollouts
        self.all_rollouts.append(rollouts)
        # Flatten the list of generations of rollouts into a list of all rollouts
        train_rollouts = [rollout for rollout_list in self.all_rollouts for rollout in rollout_list]
        print(f"Training on {len(train_rollouts)} rollouts")
        self.lstm_env = self.train(train_rollouts)