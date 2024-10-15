import gymnasium
from gymnasium import ObservationWrapper
import torch

from vae.vaemodel import VAE


class VAEWrapper(ObservationWrapper):
    """
    Wraps an environment to process observations into the latent space of a VAE.
    """
    def __init__(self, env: gymnasium.Env, vae: VAE):
        """
        TODO: Change the observation space to avoid unforeseen errors.
        """
        super(VAEWrapper, self).__init__(env)
        self.vae = vae

    def observation(self, obs):
        """
        Processes the observation through the VAE.
        Handles if the observations are batched.
        obs: (H, W, C) OR (N, H, W, C)
        z: (1, latent_dim) OR (N, latent_dim)
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        if len(obs.shape) == 3:
            obs = obs.permute(2, 0, 1)
            obs = obs.unsqueeze(0)
        else:
            obs = obs.permute(0, 3, 1, 2)
        obs /= 255.0
        mu, log_var = self.vae.encode(obs)
        z = self.vae.reparameterize(mu, log_var)
        return z
