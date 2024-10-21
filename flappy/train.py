import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from flappy.data import collect_data, Rollout, RandomPolicy
from lstm.lstmmodel import LSTM


class LSTMDataset(Dataset):
    def __init__(self, rollouts: list[Rollout], gamma: float):
        """
        Rollouts: list of rollout objects of variable length x 4 x D
        Separates the inputs, rewards, and terminateds then pads them into a single tensor.
        """

        self.lengths = torch.LongTensor([len(rollout.obses) for rollout in rollouts])

        # Preprocess inputs by concatenating obses and actions
        all_obses = [rollout.obses for rollout in rollouts]
        all_actions = [rollout.actions for rollout in rollouts]
        inputs = [torch.cat((obses, actions), dim=1) for obses, actions in zip(all_obses, all_actions)]
        self.inputs = pad_sequence(inputs, batch_first=True)

        # NOTE: We clone the rewards so we don't modify the originals
        rewards = [rollout.rewards.clone() for rollout in rollouts]
        # Multiply all the negative rewards by 100 to be stricter
        for r in rewards:
            r[r < 0] *= 100
        # Preprocess rewards by converting them to Q values with decay gamma
        rewards = [self.discount_rewards(r, gamma) for r in rewards]
        self.rewards = pad_sequence(rewards, batch_first=True)

        terminateds = [rollout.terminateds for rollout in rollouts]
        self.terminateds = pad_sequence(terminateds, batch_first=True)

    def discount_rewards(self, rewards: torch.FloatTensor, gamma: float) -> torch.FloatTensor:
        """
        Discount rewards using gamma.
        Rewards: (L, 1)
        """
        discounted = rewards.clone()
        shifteds = []
        for i in range(len(rewards)):
            shifted = torch.roll(rewards, -1 * i)
            shifted[-i:] = 0
            shifteds.append(shifted)

        for i, shifted in enumerate(shifteds):
            discounted += gamma ** i * shifted

        return discounted

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.rewards[idx], self.terminateds[idx], self.lengths[idx]


def obses_to_labels(obses: torch.FloatTensor) -> torch.FloatTensor:
    """
    obses: (N, L, D)
    Labels is obses shifted to the left by 1 in the time dimension.
    Zero out the last time step for safety.
    """
    labels = torch.roll(obses, -1, dims=1)
    labels[:, -1] = 0
    return labels


def mask_tensor(tensor: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
    """
    Masks tensor's first dimension according to lengths.
    tensor: (N, L, ...)
    lengths: (N, )
    """
    n, l = tensor.size(0), tensor.size(1)
    mask = torch.arange(l, device=tensor.device).expand(n, l) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)
    return tensor * mask.float()


def train_lstm(lstm: LSTM, dataset: Dataset, epochs: int, batch_size: int, log_dir="runs/exp", device="mps"):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lstm.train()
    lstm.to(device)
    mse = torch.nn.MSELoss()
    # TODO: Stop hard-coding the BCE loss scaling factor
    bce_scale = 100
    bce = torch.nn.BCELoss(torch.tensor([bce_scale], device=device))
    optimizer = torch.optim.AdamW(lstm.parameters())

    reward_losses = []
    obs_losses = []
    term_losses = []
    if log_dir:
        writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(epochs), desc="Training LSTM"):
        total_reward_loss = 0
        total_obs_loss = 0
        total_term_loss = 0
        for inputs, rewards, terminateds, lengths in dl:
            optimizer.zero_grad()

            inputs, rewards, terminateds, lengths = inputs.to(device), rewards.to(device), terminateds.to(device), lengths.to(device)
            # TODO: Pack the inputs to speed up LSTM time.
            # Pass inputs through LSTM. We receive padded outputs.
            pred_obses, pred_rewards, pred_terminated, _, _ = lstm(inputs)
            # Create a mask of the sequence lengths to get rid of the padding
            l_reward = mse(mask_tensor(pred_rewards, lengths), mask_tensor(rewards, lengths))
            obses = inputs[:, :, :12]
            l_obs = mse(mask_tensor(pred_obses, lengths), mask_tensor(obses_to_labels(obses), lengths))
            l_term = bce(mask_tensor(pred_terminated, lengths), mask_tensor(terminateds, lengths))

            loss = l_reward + l_obs + l_term
            loss.backward()
            optimizer.step()

            n = inputs.shape[0]
            total_reward_loss += l_reward.item() * n
            total_obs_loss += l_obs.item() * n
            total_term_loss += l_term.item() * n
        
        reward_losses.append(total_reward_loss / len(dataset))
        obs_losses.append(total_obs_loss / len(dataset))
        term_losses.append(total_term_loss / len(dataset))
        if log_dir:
            writer.add_scalar("reward loss", total_reward_loss / len(dataset), epoch)
            writer.add_scalar("z loss", total_obs_loss / len(dataset), epoch)
            writer.add_scalar("term loss", total_term_loss / len(dataset), epoch)

    return reward_losses, obs_losses, term_losses


def eval_lstm(lstm: LSTM, dataset: Dataset, device="mps"):
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    lstm.eval()
    lstm.to(device)

    mse = torch.nn.MSELoss()
    bce_scale = 100
    bce = torch.nn.BCELoss(torch.tensor([bce_scale], device=device))
    total_reward_loss = 0
    total_obs_loss = 0
    total_term_loss = 0
    with torch.no_grad():
        for inputs, rewards, terminateds, lengths in dl:

            inputs, rewards, terminateds, lengths = inputs.to(device), rewards.to(device), terminateds.to(device), lengths.to(device)

            # Pass inputs through LSTM. We receive padded outputs.
            pred_obses, pred_rewards, pred_terminated, _, _ = lstm(inputs)

            # Create a mask of the sequence lengths to get rid of the padding
            l_reward = mse(mask_tensor(pred_rewards, lengths), mask_tensor(rewards, lengths))
            obses = inputs[:, :, :12]
            l_obs = mse(mask_tensor(pred_obses, lengths), mask_tensor(obses_to_labels(obses), lengths))
            l_term = bce(mask_tensor(pred_terminated, lengths), mask_tensor(terminateds, lengths))

            n = inputs.shape[0]
            total_reward_loss += l_reward.item() * n
            total_obs_loss += l_obs.item() * n
            total_term_loss += l_term.item() * n

    print("Reward loss:", total_reward_loss / len(dataset))
    print("Z loss:", total_obs_loss / len(dataset))
    print("Term loss:", total_term_loss / len(dataset))


def test_gamma():
    policy = RandomPolicy()
    rollouts = collect_data(policy, 10000, 8, 200)

    train_rollouts = rollouts[:8000]
    test_rollouts = rollouts[8000:]

    for gamma in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"Gamma: {gamma}")
        train_ds = LSTMDataset(train_rollouts, gamma=gamma)
        lstm = LSTM(12, 1, 256)
        train_lstm(lstm, train_ds, 100, 64, log_dir=None, device="cuda")
        
        eval_ds = LSTMDataset(test_rollouts, gamma=gamma)
        eval_lstm(lstm, eval_ds, device="cuda")

def test_hidden_sizes():
    policy = RandomPolicy()
    rollouts = collect_data(policy, 10000, 8, 200)

    train_rollouts = rollouts[:8000]
    test_rollouts = rollouts[8000:]

    for hidden_size in [32, 64, 128, 256, 512]:
        print(f"hidden size: {hidden_size}")
        train_ds = LSTMDataset(train_rollouts, gamma=0.25)
        lstm = LSTM(12, 1, 256)
        train_lstm(lstm, train_ds, 100, 64, log_dir=None, device="cuda")
        
        eval_ds = LSTMDataset(test_rollouts, gamma=0.25)
        eval_lstm(lstm, eval_ds, device="cuda")

if __name__ == "__main__":
    lstm = LSTM(12, 1, 256)
    policy = RandomPolicy()
    rollouts = collect_data(policy, 10000, 8, 200)
    dataset = LSTMDataset(rollouts, gamma=0.25)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])

    reward_losses, obs_losses, term_losses = train_lstm(lstm, train_ds, 100, 64, log_dir=None, device="cuda")
    for rew, obs, term in zip(reward_losses, obs_losses, term_losses):
        print(f"Reward loss: {rew}, Z loss: {obs}, Term loss: {term}")

    torch.save(lstm.state_dict(), "flappy/lstm.pt")

    lstm = LSTM(12, 1, 256)
    lstm.load_state_dict(torch.load("flappy/lstm.pt", weights_only=True))
    eval_lstm(lstm, test_ds, device="cuda")
