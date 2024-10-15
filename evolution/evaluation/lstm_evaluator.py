import flappy_bird_gymnasium
import gymnasium
import numpy as np
import torch

from evolution.candidate import Candidate
from evolution.evaluation.evaluator import Evaluator
from lstm.lstmmodel import LSTM


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
    def __init__(self, lstm: LSTM, env: gymnasium.Env, device="cpu"):
        self.lstm = lstm
        self.lstm.to(device)
        self.lstm.eval()

        self.device = device

        self.env = env
        self.state = None
        self.h, self.c = None, None

    def reset(self, seed=None):
        """
        obses: (n_envs, D)
        self.state: (n_envs, 1, D) where 1 is T
        """
        self.h, self.c = None, None
        if seed is None:
            seed = [int(seed) for seed in np.random.randint(0, 2**32-1, self.env.num_envs)]
        obses, infos = self.env.reset(seed=seed)
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
            print(termt)
            termt = (termt > 0.5).bool()
        return self.state, rt, termt

class LSTMEvaluator(Evaluator):
    def __init__(self, n_steps=500, n_envs=32, device="cpu"):
        lstm = LSTM(12, 1, 64)
        # rollouts = collect_data(1000, 8, 100)
        # dataset = LSTMDataset(rollouts)
        # train_lstm(lstm, dataset, 100, 64)
        self.device = device
        lstm.load_state_dict(torch.load("flappy/lstm.pt", weights_only=True))
        lstm.to(device)
        lstm.eval()

        self.n_steps = n_steps
        self.n_envs = n_envs
        env = gymnasium.make_vec("FlappyBird-v0", num_envs=n_envs, use_lidar=False)
        self.lstm_env = LSTMVecEnv(lstm, env, device=device)

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
            if terminateds.any():
                print(i)
            i += 1

        print(total_rewards)
        total_reward = total_rewards.mean().item()
        print(total_reward)
        candidate.metrics["reward"] = total_reward
        assert False
