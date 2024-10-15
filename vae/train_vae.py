"""
Standard PyTorch training code for the VAE.
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from vae.vaemodel import VAE


class ImageDataset(Dataset):
    """
    Custom PyTorch dataset preprocessing our images into the format Torch wants
    """
    def __init__(self, img_data: np.ndarray):
        """
        img_data: (N, H, W, C) array of images
        """
        self.data = img_data
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data /= 255  # Normalize to [0, 1]
        self.data = self.data.permute(0, 3, 1, 2)  # Go from NHWC to NCHW

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(0, dtype=torch.float32)


def train_model(vae: VAE, train_ds: Dataset, epochs: int, batch_size: int, device="mps"):
    """
    Trains a VAE model on the given dataset.
    """
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    vae.to(device)
    optimizer = torch.optim.AdamW(vae.parameters())
    vae.train()
    avg_recons = []
    avg_klds = []
    for epoch in range(epochs):
        reconses = []
        klds = []
        for x, _ in train_dl:
            optimizer.zero_grad()
            x = x.to(device)
            xhat, x, mu, log_var = vae(x)
            recons, kld = vae.loss_function(xhat, x, mu, log_var)
            loss = recons + kld
            loss.backward()
            optimizer.step()

            reconses.append(recons.item())
            klds.append(kld.item())

        avg_recon = sum(reconses) / len(reconses)
        avg_kld = sum(klds) / len(klds)
        avg_recons.append(avg_recon)
        avg_klds.append(avg_kld)
        print(f"Epoch {epoch + 1}/{epochs} | Recons: {avg_recon} | KLD: {avg_kld}")

    return avg_recons, avg_klds


def main():
    """
    Main training logic for the VAE.
    """
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = "smallhumanvae"

    vae = VAE(config["img_size"], config["latent_dim"], config["encoder_blocks"], config["decoder_blocks"])

    img_data = np.load("data/CarRacing-v2/human.npy")
    dataset = ImageDataset(img_data)

    recons, klds = train_model(vae, dataset, 10, 64)
    torch.save(vae.state_dict(), f"models/{model_name}.pt")
    results = pd.DataFrame({"recons": recons, "klds": klds})
    results.to_csv(f"models/{model_name}_results.csv")


if __name__ == "__main__":
    main()
