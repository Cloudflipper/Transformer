import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset,TensorDataset
import os
import math
import logging

class DynamicPositionalEncoding(nn.Module):
    """
    A PyTorch module that dynamically generates positional encoding for sequences of arbitrary length.
    The formula for the embeddings is:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Attribute:
        d_model (int): The dimensionality of the input embeddings.
    Method:
        forward(x): Dynamically generates and adds positional encoding to the input embeddings.

    Example:
        >>> pos_encoder = DynamicPositionalEncoding(d_model=512)
        >>> input_embeddings = torch.randn(100, 32, 512)  
        >>> encoded_embeddings = pos_encoder(input_embeddings)
        >>> print(encoded_embeddings.shape)
        torch.Size([100, 32, 512])
    """

    def __init__(self, d_model=512):
        """
        Arg:
            d_model (int): The dimensionality of the input embeddings.
        """
        super(DynamicPositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        Dynamically generates and adds positional encoding to the input embeddings.

        Arg:
            x (torch.Tensor): The input embeddings with shape (seq_len, batch_size, d_model).
        Returns:
            torch.Tensor: The input embeddings with position encodings added. Shape (seq_len, batch_size, d_model).
        """
        seq_len, batch_size, d_model = x.size()
        position=torch.arange(seq_len, device=x.device).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * (-math.log(1000.0) / d_model))  

        pe_sin,pe_cos = torch.sin(position * div_term),torch.cos(position * div_term)  
        pe = torch.zeros(seq_len, d_model, device=x.device) 
        pe[:, 0::2] = pe_sin  
        pe[:, 1::2] = pe_cos  

        return x + pe.unsqueeze(1) 

def he_init(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class OnlineTransformer(nn.Module):
    """
    An Online Transformer-based model for sequence reconstruction tasks.

    This model processes input sequences one data point at a time, maintaining a memory buffer
    to store previous inputs. It leverages the Transformer encoder to encode the input sequence,
    applies a Diffusion Model to enhance the encoded representation, and then uses a Transformer
    decoder to reconstruct the input sequence.

    Attributes:
        d_model (int): The dimensionality of the model's embeddings and FEATURES. In this task it is same as input_dim to reduce lineaar layers.
        nhead (int): The number of attention heads in the Transformer.
        num_encoder_layers (int): The number of layers in the Transformer encoder.
        num_decoder_layers (int): The number of layers in the Transformer decoder.
        dim_feedforward (int): The dimensionality of the feedforward network in the Transformer.
        dropout (float): The dropout probability.
        memory_size (int): The size of the memory buffer to store previous inputs.
        positional_encoding (DynamicPositionalEncoding): Module to add dynamic positional encoding to the input embeddings.
        transformer_encoder (nn.TransformerEncoder): The Transformer encoder.
        transformer_decoder (nn.TransformerDecoder): The Transformer decoder.
        output_layer (nn.Linear): Linear layer to project the Transformer output back to the input dimension.
        memory (torch.Tensor): The memory buffer to store previous input embeddings.
    """
    def __init__(self, input_dim,output_dim,d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward=128, dropout=0.1, memory_size=100,batch_size=32, device = "cuda"):
        super(OnlineTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.memory_size = memory_size
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = DynamicPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.memory = torch.zeros(memory_size, batch_size, d_model).to(device)
        self.apply(he_init)

    
    def forward(self, x):
        """
        Processes a new input data point and generates a reconstructed output based on the current input and memory.
        Args:
            x (torch.Tensor): The new input data point with shape (batch_size, input_dim).
        Returns:
            torch.Tensor: The reconstructed output with shape (batch_size, input_dim).
        """
        x = self.input_embedding(x)  # (batch_size, d_model)
        x = self.positional_encoding(x.unsqueeze(0)).squeeze(0)# (batch_size, d_model)

        self.update_memory(x)
        input_seq = torch.cat([self.memory, x.unsqueeze(0)], dim=0)  # (memory_size + 1, batch_size, d_model)
        feature = self.transformer_encoder(input_seq)  # (memory_size + 1, batch_size, d_model)
        self.feature = feature.clone()
        # if self.diffusion_model is not None:
        #     encoder_output = encoder_output.permute(1, 0, 2)  # (batch_size, memory_size + 1, d_model)
        #     encoder_output = self.diffusion_model(encoder_output) 
        #     encoder_output = encoder_output.permute(1, 0, 2)
        decoder_output = self.transformer_decoder(feature, feature)  # (memory_size + 1, batch_size, d_model)
        output = decoder_output[-1, :, :]  # (batch_size, d_model)
        reconstructed = self.output_layer(output)  # (batch_size, input_dim)
        return reconstructed

    def update_memory(self, x):
        """
        Updates the memory buffer with the new input embedding.
        Args:
            x (torch.Tensor): The new input embedding with shape (batch_size, d_model).
        """
        x = x.unsqueeze(0)  # (1, batch_size, d_model)
        # print(self.memory.shape)
        # print(x.shape)
        self.memory = torch.cat([self.memory[1:], x], dim=0)  # (memory_size, batch_size, d_model)
        
    def get_feature(self):
        return self.feature
    
    def train_all(self,priviledge, obs_with_noise,batch_size=32, epochs=10000, 
              learning_rate=1e-3, validation_split=0.2, save_dir="saved_models"):
        """
        Trains the model using the given training data.
        Args:
            priviledge (torch.Tensor): The priviledge data with shape (batch_size, input_dim).
            obs_with_noise (torch.Tensor): The observed data with noise with shape (batch_size, input_dim).
            batch_size (int): The batch size for training.
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for the optimizer.
            validation_split (float): The proportion of the dataset to include in the validation split.
        """
        logging.basicConfig(filename=os.path.join(save_dir, 'training.log'),
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        self.to(device)
        priviledge_tensor = torch.tensor(priviledge[:16000], dtype=torch.float32).to(device)
        obs_with_noise_tensor = torch.tensor(obs_with_noise[:16000], dtype=torch.float32).to(device)

        # Create dataset and data loader
        dataset = TensorDataset(obs_with_noise_tensor, priviledge_tensor)
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define optimizer and loss function
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(epochs):
            #self.train(priviledge_tensor , obs_with_noise_tensor)
            train_loss = 0.0
            for inputs, targets in train_loader:
                #print(inputs.shape)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward(retain_graph=True)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}")
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}")

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}")
            logging.info(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}")
            
            #save model every 100 steps
            if (epoch+1) % 100 == 0:
                checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")
                logging.info(f"Model checkpoint saved to {checkpoint_path}")
        
    
# class DiffusionModel(nn.Module):
#     """
#     A Diffusion Model for progressively adding and removing noise from data.

#     This model leverages a Transformer-based model to predict noise in the data,
#     allowing it to reconstruct the original data from a noisy version. The diffusion
#     process consists of two main stages:
#     1. Forward Process (q_sample): Gradually adds noise to the data over a series of timesteps.
#     2. Reverse Process (p_sample): Uses the Transformer model to predict and remove noise,
#        reconstructing the original data.

#     Attributes:
#         transformer_model (nn.Module): The Transformer model used to predict noise.
#         timesteps (int): The number of timesteps in the diffusion process.
#         betas (torch.Tensor): A tensor of noise coefficients for each timestep.
#         alphas (torch.Tensor): A tensor of coefficients representing 1 - betas.
#         alpha_bar (torch.Tensor): A tensor of cumulative products of alphas.

#     Methods:
#         q_sample(x_0, t, noise): Applies the forward diffusion process to add noise to the data.
#         p_sample(x_t, t, cond): Applies the reverse diffusion process to remove noise from the data.
#         forward(x_0, cond): Trains the model by predicting noise for a given timestep.
#     """
#     def __init__(self, transformer_model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
#         super(DiffusionModel, self).__init__()
#         self.transformer_model = transformer_model
#         self.timesteps = timesteps
#         self.betas = torch.linspace(beta_start, beta_end, timesteps)
#         self.alphas = 1 - self.betas
#         self.alpha_bar = torch.cumprod(self.alphas, dim=0)

#     def q_sample(self, x_0, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_0)
#         sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t])[:, None, None]
#         sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t])[:, None, None]
#         return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    
#     def p_sample(self, x_t, t):
#         noise_pred = self.transformer_model(x_t)
#         sqrt_alpha_t = torch.sqrt(self.alphas[t])[:, None, None]
#         sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas[t])[:, None, None]
#         return (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

#     def forward(self, x_0):
#         """
#         Trains the diffusion model by predicting the noise added to the data.

#         Args:
#             x_0 (torch.Tensor): The clean data with shape (batch_size, seq_len, feature_dim).
#             cond (torch.Tensor): The conditional sequence with shape (batch_size, seq_len, feature_dim).

#         Returns:
#             torch.Tensor: The predicted noise with shape (batch_size, seq_len, feature_dim).
#         """
#         t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device)
#         noise = torch.randn_like(x_0)
#         x_t = self.q_sample(x_0, t, noise)
#         noise_pred = self.transformer_model( x_t)
#         return noise_pred


def add_segmented_noise_to_tensor(input_array, segment_lengths, noise_stds):
    import numpy as np
    """
    Add different Gaussian noise to different segments of the input array.

    Parameters:
        input_array: The input array with shape (batch_size, input_size).
        segment_lengths: A list of integers specifying the lengths of each segment.
                         The sum of segment_lengths must equal input_size.
        noise_stds: The standard deviation of the noise for each segment.
    Returns:
        The array with added noise in different segments.
    Examples:
        >>> batch_size = 4
        >>> input_size = 10
        >>> input_tensor = np.random.randn(batch_size, input_size)
        >>> segment_lengths = [3, 4, 3]  # The sum of these lengths must equal input_size (10 in this case)
        >>> noise_stds = [0.1, 0.2, 0.3]  # Different noise standard deviations for each segment
        >>> noisy_tensor = add_segmented_noise_to_tensor(input_tensor, segment_lengths, noise_stds)
    """

    if not isinstance(input_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if sum(segment_lengths) != input_array.shape[1]:
        raise ValueError("The sum of segment_lengths must equal input_size")
    
    start_idx = 0
    noisy_segments = []
    for length, noise_std in zip(segment_lengths, noise_stds):
        end_idx = start_idx + length
        segment = input_array[:, start_idx:end_idx]
        noise = np.random.randn(*segment.shape) * noise_std
        noisy_segment = segment + noise
        noisy_segments.append(noisy_segment)
        start_idx = end_idx
    
    noisy_array = np.concatenate(noisy_segments, axis=1)
    
    return noisy_array



if __name__ == '__main__':
    model = OnlineTransformer(input_dim=45,output_dim=12, d_model=512, nhead=8, 
                                         num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)

    x = torch.randn(32, 6)  # (batch_size, input_dim)

    reconstructed = model(x)

    print("Reconstructed shape:", reconstructed.shape)  # (batch_size, input_dim)