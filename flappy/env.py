import flappy_bird_gymnasium
import gymnasium
import numpy as np
import torch

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
    

def main():
    env = gymnasium.make("FlappyBird-v0", use_lidar=False)
    lstm = LSTM(12, 1, 256)
    lstm.load_state_dict(torch.load("flappy/lstm.pt", weights_only=True))
    lstm_vec_env = LSTMVecEnv(lstm, env, 1, device="cuda")
    
    obs = lstm_vec_env.reset()
    done = False
    total_reward = 0
    while not done:
        print(obs)
        action = input("Enter action:")
        action = torch.LongTensor([[[int(action)]]]).to("cuda")
        obs, reward, done = lstm_vec_env.step(action)
        print(reward, done)
        total_reward += reward.item()
        print(total_reward)
        done = done.item()


if __name__ == "__main__":
    main()