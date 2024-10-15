import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from flappy.data import collect_data
from lstm.lstmmodel import LSTM


class LSTMDataset(Dataset):
    def __init__(self, rollouts):
        """
        Rollouts: (N, L, 4) with L varying in length
        """
        # Split rollouts into obses, actions, rewards, next_obses
        obses = []
        actions = []
        rewards = []
        terminateds = []
        next_obses = []

        for rollout in rollouts:
            roll_obs = []
            roll_act = []
            roll_rew = []
            roll_ter = []
            roll_nex = []
            for obs, act, rew, ter, nex in rollout:
                roll_obs.append(obs)
                roll_act.append(act)
                roll_rew.append(rew)
                roll_ter.append(ter)
                roll_nex.append(nex)

            obses.append(torch.tensor(np.array(roll_obs), dtype=torch.float32))
            actions.append(torch.tensor(np.array(roll_act), dtype=torch.float32))
            rewards.append(torch.tensor(np.array(roll_rew), dtype=torch.float32))
            terminateds.append(torch.tensor(np.array(roll_ter), dtype=torch.float32))
            next_obses.append(torch.tensor(np.array(roll_nex), dtype=torch.float32))

        self.inputs = [torch.cat((obs, act), dim=1) for obs, act in zip(obses, actions)]
        self.rewards = rewards
        self.terminateds = terminateds
        self.next_obses = next_obses

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.rewards[idx], self.terminateds[idx], self.next_obses[idx]

def collate_fn(batch):
    inputs, rewards, terminateds, next_obses = zip(*batch)

    lens = [len(inp) for inp in inputs]
    padding_value = 0
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_value)
    rewards = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True, padding_value=padding_value)
    terminateds = torch.nn.utils.rnn.pad_sequence(terminateds, batch_first=True, padding_value=padding_value)
    next_obses = torch.nn.utils.rnn.pad_sequence(next_obses, batch_first=True, padding_value=padding_value)

    lens, sorted_idx = torch.tensor(lens).sort(descending=True)
    inputs = inputs[sorted_idx]
    rewards = rewards[sorted_idx]
    terminateds = terminateds[sorted_idx]
    next_obses = next_obses[sorted_idx]

    inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lens, batch_first=True)
    rewards = torch.nn.utils.rnn.pack_padded_sequence(rewards, lens, batch_first=True)
    terminateds = torch.nn.utils.rnn.pack_padded_sequence(terminateds, lens, batch_first=True)
    next_obses = torch.nn.utils.rnn.pack_padded_sequence(next_obses, lens, batch_first=True)

    return inputs, rewards, terminateds, next_obses, lens


def train_lstm(lstm: LSTM, dataset: Dataset, epochs: int, batch_size: int, log_dir="runs/exp", device="mps"):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    lstm.train()
    lstm.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(lstm.parameters())
    writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(epochs)):
        total_reward_loss = 0
        total_z_loss = 0
        total_term_loss = 0
        for inp, r, term, next_z, lens in dl:
            optimizer.zero_grad()

            inp, r, term, next_z = inp.to(device), r.to(device), term.to(device), next_z.to(device)

            zt, rt, termt, ht, ct = lstm(inp)

            r, _ = torch.nn.utils.rnn.pad_packed_sequence(r, batch_first=True, padding_value=0)
            next_z, _ = torch.nn.utils.rnn.pad_packed_sequence(next_z, batch_first=True, padding_value=0)
            term, _ = torch.nn.utils.rnn.pad_packed_sequence(term, batch_first=True, padding_value=0)

            l_reward = criterion(rt, r)
            l_z = criterion(zt, next_z)
            l_term = criterion(termt, term)
            loss = l_reward + l_z + l_term
            loss.backward()
            optimizer.step()

            total_reward_loss += l_reward.item() * zt.shape[0]
            total_z_loss += l_z.item() * zt.shape[0]
            total_term_loss += l_term.item() * zt.shape[0]
        
        writer.add_scalar("reward loss", total_reward_loss / len(dataset), epoch)
        writer.add_scalar("z loss", total_z_loss / len(dataset), epoch)
        writer.add_scalar("term loss", total_term_loss / len(dataset), epoch)

def eval_lstm(lstm: LSTM, dataset: Dataset, device="mps"):
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    lstm.eval()
    lstm.to(device)

    criterion = torch.nn.MSELoss()
    total_reward_loss = 0
    total_z_loss = 0
    total_term_loss = 0
    with torch.no_grad():
        for inp, r, term, next_z in dl:
            inp, r, term, next_z = inp.to(device), r.to(device), term.to(device), next_z.to(device)

            zt, rt, termt, ht, ct = lstm(inp)

            l_reward = criterion(rt, r)
            l_z = criterion(zt, next_z)
            l_term = criterion((termt > 0.5).float(), term)

            total_reward_loss += l_reward.item()
            total_z_loss += l_z.item()
            total_term_loss += l_term.item()

    print("Reward loss:", total_reward_loss / len(dataset))
    print("Z loss:", total_z_loss / len(dataset))
    print("Term loss:", total_term_loss / len(dataset))


if __name__ == "__main__":
    lstm = LSTM(12, 1, 64)
    rollouts = collect_data(1000, 8, 500)
    dataset = LSTMDataset(rollouts)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_lstm(lstm, train_ds, 100, 64, log_dir="runs/longerdata")

    torch.save(lstm.state_dict(), "flappy/lstm.pt")

    lstm = LSTM(12, 1, 64)
    lstm.load_state_dict(torch.load("flappy/lstm.pt", weights_only=True))
    eval_lstm(lstm, test_ds)
