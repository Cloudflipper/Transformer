import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    """
    A Diffusion Model for progressively adding and removing noise from data.

    This model leverages a Transformer-based model to predict noise in the data,
    allowing it to reconstruct the original data from a noisy version. The diffusion
    process consists of two main stages:
    1. Forward Process (q_sample): Gradually adds noise to the data over a series of timesteps.
    2. Reverse Process (p_sample): Uses the Transformer model to predict and remove noise,
       reconstructing the original data.

    Attributes:
        transformer_model (nn.Module): The Transformer model used to predict noise.
        timesteps (int): The number of timesteps in the diffusion process.
        betas (torch.Tensor): A tensor of noise coefficients for each timestep.
        alphas (torch.Tensor): A tensor of coefficients representing 1 - betas.
        alpha_bar (torch.Tensor): A tensor of cumulative products of alphas.

    Methods:
        q_sample(x_0, t, noise): Applies the forward diffusion process to add noise to the data.
        p_sample(x_t, t, cond): Applies the reverse diffusion process to remove noise from the data.
        forward(x_0, cond): Trains the model by predicting noise for a given timestep.
    """
    def __init__(self, transformer_model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.transformer_model = transformer_model
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t])[:, None, None]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t])[:, None, None]
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    
    def p_sample(self, x_t, t):
        noise_pred = self.transformer_model(x_t)
        sqrt_alpha_t = torch.sqrt(self.alphas[t])[:, None, None]
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas[t])[:, None, None]
        return (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

    def forward(self, x_0):
        """
        Trains the diffusion model by predicting the noise added to the data.

        Args:
            x_0 (torch.Tensor): The clean data with shape (batch_size, seq_len, feature_dim).
            cond (torch.Tensor): The conditional sequence with shape (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: The predicted noise with shape (batch_size, seq_len, feature_dim).
        """
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self.transformer_model( x_t)
        return noise_pred