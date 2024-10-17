import json

import gymnasium
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vae.vaemodel import VAE
from vae.collect_data import Rollout


class LSTM(torch.nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(latent_dim + action_dim, hidden_size, batch_first=True)
        self.reward_fc = torch.nn.Linear(hidden_size, 1)
        self.term_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )
        self.latent_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * 2, latent_dim)
        )

    def forward(self, inp: torch.Tensor, h=None, c=None) -> torch.Tensor:
        """
        z: (N, latent_dim)
        a: (N, action_dim)
        """
        if h is not None and c is not None:
            hout, (ht, ct) = self.lstm(inp, (h, c))
        else:
            hout, (ht, ct) = self.lstm(inp)

        # Once we have the output we need to unpack it to pass through the linear layers
        if isinstance(hout, torch.nn.utils.rnn.PackedSequence):
            hout, _ = torch.nn.utils.rnn.pad_packed_sequence(hout, batch_first=True)
        
        reward = self.reward_fc(hout)
        zt = self.latent_fc(hout)
        termt = self.term_fc(hout)
        return zt, reward, termt, ht, ct

class VAEDS(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        data: (4, N, L, D)
        obses: (N, H, W, C)
        x: (N, C, H, W)
        """
        obses = torch.tensor(data[0], dtype=torch.float32)
        obses = obses.reshape(-1, *obses.shape[-3:])
        obses = obses.permute(0, 3, 1, 2)
        obses /= 255.0
        self.x = obses
        self.y = torch.zeros_like(self.x)
        print("VAE data shape: ", self.x.shape)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LSTMDS(torch.utils.data.Dataset):

    def vae_process_imgs(self, data: list, vae: VAE, batch_size=64, device="mps") -> torch.Tensor:
        """
        data: (4, N, L, H, W, C)
        x: (N*L, C, H, W)
        zs: (N, L, latent_dim)
        """
        vae.to(device)
        vae.eval()
        with torch.no_grad():
            l = data[0].shape[1]
            vae_dataset = VAEDS(data)
            dl = DataLoader(vae_dataset, batch_size=batch_size, shuffle=False)
            zs = []
            for x, _ in dl:
                x = x.to(device)
                mu, log_var = vae.encode(x)
                z = vae.reparameterize(mu, log_var)
                zs.append(z.cpu())

            zs = torch.cat(zs, dim=0)
            zs = zs.reshape(-1, l, zs.shape[-1])
        return zs

    def __init__(self, rollouts: list[Rollout], vae: VAE):
        """
        data: (4, N, L, D)
        z: (N, L, C, H, W)
        a: (N, L, 1)
        r: (N, L, 1)
        next_z: (N, L, C, H, W)
        """
        obses = [torch.tensor(rollout.obses) for rollout in rollouts]
        actions = [torch.tensor(rollout.actions) for rollout in rollouts]
        rewards = [torch.tensor(rollout.rewards) for rollout in rollouts]
        next_obses = [torch.tensor(rollout.next_obses) for rollout in rollouts]

        len_obses = torch.tensor([obs.shape[0] for obs in obses])
        obses = torch.nn.utils.rnn.pad_sequence(obses, batch_first=True)
        len_actions = torch.tensor([action.shape[0] for action in actions])
        actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True)
        len_rewards = torch.tensor([reward.shape[0] for reward in rewards])
        rewards = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True)
        len_next_obses = torch.tensor([next_obs.shape[0] for next_obs in next_obses])
        next_obses = torch.nn.utils.rnn.pad_sequence(next_obses, batch_first=True)

        self.z = self.vae_process_imgs(data, vae).detach().clone()
        self.a = torch.tensor(data[1], dtype=torch.float32).detach().clone()
        self.r = torch.tensor(data[2], dtype=torch.float32).detach().clone()
        next_data = [data[3], None, None, None]
        self.next_z = self.vae_process_imgs(next_data, vae).detach().clone()
        print("LSTM Data shape")
        print(self.z.shape, self.a.shape, self.r.shape, self.next_z.shape)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.a[idx], self.r[idx], self.next_z[idx]


def train_lstm(lstm: LSTM, dataset: Dataset, epochs: int, batch_size: int, log_dir="runs/exp", device="mps"):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lstm.train()
    lstm.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(lstm.parameters())
    writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(epochs)):
        total_reward_loss = 0
        total_z_loss = 0
        for z, a, r, next_z in dl:
            optimizer.zero_grad()

            z, a, r, next_z = z.to(device), a.to(device), r.to(device), next_z.to(device)

            h0 = torch.zeros(1, z.shape[0], 64).detach().clone().to(device)
            c0 = torch.zeros(1, z.shape[0], 64).detach().clone().to(device)
            zt, rt, ht, ct = lstm(z, a, h0, c0)

            l_reward = criterion(rt, r)
            l_z = criterion(zt, next_z)
            loss = l_reward + l_z
            loss.backward()
            optimizer.step()

            total_reward_loss += l_reward.item() * z.shape[0]
            total_z_loss += l_z.item() * z.shape[0]
        
        writer.add_scalar("reward loss", total_reward_loss / len(dataset), epoch)
        writer.add_scalar("z loss", total_z_loss / len(dataset), epoch)
