import flappy_bird_gymnasium
import gymnasium
import numpy as np
import torch

from evolution.candidate import Candidate
from evolution.evaluation.evaluator import Evaluator
from lstm.lstmmodel import LSTM
from flappy.data import CandidatePolicy, collect_data
from flappy.train import train_lstm, LSTMDataset


class LSTMEnv:
    def __init__(self, lstm: LSTM, env: gymnasium.Env, device="cpu"):
        self.lstm = lstm
        self.lstm.to(device)
        self.lstm.eval()

        self.device = device

        self.env = env  # Used to handle initial state
        self.state = None
        self.h, self.c = None, None

    def reset(self, seed=None):
        self.h, self.c = None, None
        obs, info = self.env.reset(seed=seed)
        self.state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)
        return self.state

    def step(self, action):
        inp = torch.cat((self.state, action), dim=2)
        self.state, rt, termt, self.h, self.c = self.lstm(inp, self.h, self.c)

        termt = (termt > 0.5).bool().item()

        return self.state, rt, termt
    
class LSTMVecEnv:
    """
    'Vectorized' version of the LSTM Env. Just batches the observations through the LSTM.
    """
    def __init__(self, lstm: LSTM, env: gymnasium.Env, num_envs: int, device="cpu"):
        self.lstm = lstm
        self.lstm.to(device)
        self.lstm.eval()

        self.device = device

        self.n_envs = num_envs
        self.env = env
        self.state = None
        self.h, self.c = None, None

    def reset(self, seed=None):
        """
        obses: (n_envs, D)
        self.state: (n_envs, 1, D) where 1 is T
        """
        if seed is None:
            seed = np.random.randint(0, 2**32-1, self.n_envs)

        self.h, self.c = None, None
        obses = []
        for s in seed:
            obs, info = self.env.reset(seed=int(s))
            obses.append(obs)
        obses = np.array(obses)
        self.state = torch.tensor(obses, dtype=torch.float32).unsqueeze(1).to(self.device)
        return self.state
    
    def step(self, actions):
        """
        actions: (n_envs, 1, 1)
        inp: (n_envs, 1, D+1)
        """
        with torch.no_grad():
            inp = torch.cat([self.state, actions], dim=2)
            self.state, rt, termt, self.h, self.c = self.lstm(inp, self.h, self.c)
            termt = (termt > 0.5).bool()
        return self.state, rt, termt

class LSTMEvaluator(Evaluator):
    """
    Evaluates candidates in the LSTM world model.
    We draw a sample from the true env and use it to start the LSTM.
    """
    def __init__(self, n_steps=500, n_envs=32, device="cpu", log_path=None):
        lstm = LSTM(12, 1, 256)
        # rollouts = collect_data(1000, 8, 100)
        # dataset = LSTMDataset(rollouts)
        # train_lstm(lstm, dataset, 100, 64)
        self.device = device
        lstm.load_state_dict(torch.load("flappy/fixedlstm.pt", weights_only=True))
        lstm.to(device)
        lstm.eval()

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.env = gymnasium.make("FlappyBird-v0", use_lidar=False)
        self.lstm_env = LSTMVecEnv(lstm, self.env, num_envs=self.n_envs, device=device)

        self.log_path = log_path

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

    def retrain_lstm(self, candidates: list[Candidate], n_rollouts: int, n_steps: int, n_envs: int):
        assert n_rollouts % len(candidates) == 0
        all_rollouts = []
        for candidate in candidates:
            policy = CandidatePolicy(candidate)
            rollouts = collect_data(policy, n_rollouts // len(candidates), n_envs=n_envs, n_steps=n_steps)
            all_rollouts.extend(rollouts)

        # Log the rewards of the collected rollouts
        if self.log_path:
            rewards = np.array([np.mean(rollout.rewards) for rollout in all_rollouts])
            rewards = rewards.view(len(candidates), n_rollouts // len(candidates))
            cand_rewards = np.mean(rewards, axis=1)
            cand_rewards = cand_rewards.tolist()
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{cand_rewards}\n")

        # TODO: un-hardcode gamma
        dataset = LSTMDataset(all_rollouts, gamma=0.1)
        new_lstm = LSTM(12, 1, 256)
        train_lstm(new_lstm, dataset, 100, 64, None, device="mps")
        new_lstm.to(self.device)
        new_lstm.eval()
        self.lstm_env = LSTMVecEnv(new_lstm, self.env, self.n_envs, device=self.device)