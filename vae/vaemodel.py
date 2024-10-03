"""
The VAE class we use to encode the frames of our game.
"""
import torch
import torch.nn.functional as F


class VAE(torch.nn.Module):
    """
    Variational AutoEncoder model.
    """
    @staticmethod
    def create_encoder_block(in_channels, out_channels, kernel_size, stride, padding=0):
        """
        Creates down convolution block with the specified params.
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.ReLU()
        )

    @staticmethod
    def conv_output_size(img_size, encoder_params):
        """
        Computes the encoded image size after we pass it through the encoder.
        """
        size = img_size
        for block_params in encoder_params:
            kernel_size = block_params.get("kernel_size")
            stride = block_params.get("stride", 1)
            padding = block_params.get("padding", 0)
            size = ((size - kernel_size + 2 * padding) // stride) + 1
        return size

    @staticmethod
    def create_decoder_block(in_channels, out_channels, kernel_size, stride, padding=0, activation="relu"):
        """
        Creates up convolution block with the specified params.
        """
        act = torch.nn.ReLU() if activation == "relu" else torch.nn.Sigmoid()
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            act
        )

    def __init__(self, img_size, latent_dim, encoder_params, decoder_params):
        super(VAE, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim

        # Encoder
        encoder_blocks = []
        for block_params in encoder_params:
            encoder_blocks.append(self.create_encoder_block(**block_params))
        self.encoder = torch.nn.Sequential(*encoder_blocks)

        self.encoded_img_size = self.conv_output_size(img_size, encoder_params)

        # Latent space
        self.final_hid = encoder_params[-1]["out_channels"]
        self.fc_mu = torch.nn.Linear((self.encoded_img_size ** 2) * self.final_hid, self.latent_dim)
        self.fc_var = torch.nn.Linear((self.encoded_img_size ** 2) * self.final_hid, self.latent_dim)

        # Decoder
        self.decoder_input = torch.nn.Linear(self.latent_dim, self.encoded_img_size ** 2 * self.final_hid)
        decoder_blocks = []
        for block_params in decoder_params:
            decoder_blocks.append(self.create_decoder_block(**block_params))
        self.decoder = torch.nn.Sequential(*decoder_blocks)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes x into mu and log_var to be used for reparameterization.
        x: (N, C, H, W)
        mu: (N, latent_dim)
        log_var: (N, latent_dim)
        """
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0, 1).
        mu: (N, latent_dim)
        log_var: (N, latent_dim)
        z: (N, latent_dim)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes from latent space to the original space.
        z: (N, latent_dim)
        xhat: (N, C, H, W)
        """
        xhat = self.decoder_input(z)
        xhat = xhat.reshape(xhat.shape[0], self.final_hid, self.encoded_img_size, self.encoded_img_size)
        xhat = self.decoder(xhat)
        return xhat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass of our model returning the regenerated image.
        Additionally returns the original label, mu, and log_var for the loss function.
        x: (N, C, H, W)
        xhat: (N, C, H, W)
        mu: (N, latent_dim)
        log_var: (N, latent_dim)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        xhat = self.decode(z)
        if len(xhat.shape) == 3:
            xhat = xhat.unsqueeze(1)
        return xhat, x, mu, log_var

    def generate(self, n_samples: int) -> torch.Tensor:
        """
        Randomly samples from the latent space and decodes it.
        """
        z = torch.randn(n_samples, self.latent_dim)
        x_hat = self.decode(z)
        return x_hat

    @staticmethod
    def loss_function(xhat: torch.Tensor,
                      x: torch.Tensor,
                      mu: torch.Tensor,
                      log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the VAE loss function.
        NOTE: We should probably put in a weight for the KLD loss.
        """
        xhat_flat = xhat.view(xhat.shape[0], -1)
        x_flat = x.view(x.shape[0], -1)

        recons_loss = F.mse_loss(xhat_flat, x_flat, reduction="sum") / xhat_flat.shape[0]

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim=1), dim=0)

        return recons_loss, kld_loss
