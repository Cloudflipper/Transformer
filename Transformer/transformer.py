import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset,TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import math
import logging
import time

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

    def __init__(self, d_model=512,max_len=5000, device=None):
        """
        Arg:
            d_model (int): The dimensionality of the input embeddings.
        """
        super(DynamicPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 不需要计算梯度

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 计算位置编码
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))


    def forward(self, x):
        """
        Dynamically generates and adds positional encoding to the input embeddings.

        Arg:
            x (torch.Tensor): The input embeddings with shape (seq_len, batch_size, d_model).
        Returns:
            torch.Tensor: The input embeddings with position encodings added. Shape (seq_len, batch_size, d_model).
        """
        #print("x.shape",x.shape)
        
        batch_size, seq_len, dim = x.size()
        #print("self.encoding.shape",self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)[0,1,:])
        return self.encoding[:seq_len, :dim].unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)+x


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
                 dim_feedforward=128, dropout=0.1, memory_size=100,batch_size=32, device = "cuda:2",
                 mean = None, std = None):
        super(OnlineTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.memory_size = memory_size
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.batch_size = batch_size
        
        self.pre_layer = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.positional_encoding = DynamicPositionalEncoding(d_model,device=device)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )
        self.mean = mean
        self.std = std
        self.apply(he_init)
        

    
    def forward(self, x):
        """
        Processes a new input data point and generates a reconstructed output based on the current input and memory.
        Args:
            x (torch.Tensor): The new input data point with shape (batch_size, input_dim).
        Returns:
            torch.Tensor: The reconstructed output with shape (batch_size, input_dim).
        """
        #print("x.shape",x.shape)
        res = self.pre_layer(x)
        res = self.positional_encoding(res).squeeze(0)# (batch_size, seq_length, d_model)
        
        #print("x.shape2",x.shape)
        
        #print("x.shape3",x.shape)
        #input_seq = torch.cat([self.memory, x.unsqueeze(0)], dim=0)  # (memory_size + 1, batch_size, d_model)
        res = res.permute(1, 0, 2)
        
        feature = self.transformer_encoder(res)  # (seq_length, batch_size, d_model)
        self.feature = torch.cat((feature.permute(1, 0, 2),x),dim=2)
        # if self.diffusion_model is not None:
        #     encoder_output = encoder_output.permute(1, 0, 2)  # (batch_size, memory_size + 1, d_model)
        #     encoder_output = self.diffusion_model(encoder_output) 
        #     encoder_output = encoder_output.permute(1, 0, 2)
        decoder_output = self.transformer_decoder(feature, feature)  # (seq_length, batch_size, d_model)
        #print("decoder_output.shape",decoder_output.shape)
        reconstructed = self.output_layer(decoder_output)  # (batch_size, input_dim)
        reconstructed = reconstructed.permute(1, 0, 2)
        #print("reconstructed.shape",reconstructed.shape)
        return reconstructed

        
    def get_feature(self):
        return self.feature
    
    def train_all(self,priviledge, obs_with_noise,batch_size=32, epochs=10000, trajectory = 128,
              learning_rate=1e-3, validation_split=0.2, save_dir="saved_models",save_model = True):
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
        logging.info(f"self.d_model:{self.d_model}, self.n_head: {self.nhead}, self.num_encoder_layers: {self.num_encoder_layers}, self.num_decoder_layers: {self.num_decoder_layers}, \
                     self.dim_feedforward: {self.dim_feedforward}, self.dropout: {self.dropout}, self.memory_size: {self.memory_size}")
        
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.device = device
        #device = torch.device("cpu")
        self.to(device)
        priviledge_tensor = torch.tensor(priviledge, dtype=torch.float32).to(device)
        obs_with_noise_tensor = torch.tensor(obs_with_noise, dtype=torch.float32).to(device)

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
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,          
            T_mult=2,        
            eta_min=1e-4     
        )

        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            #self.train(priviledge_tensor , obs_with_noise_tensor)
            train_loss = 0.0
            train_start_time = time.time()
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                output_t = self(inputs)  
                total_loss = criterion(output_t, targets)
                total_loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(train_loader))
                train_loss += total_loss.item()

            train_loss /= len(train_loader.dataset)
            train_end_time = time.time()

            # Validation
            self.eval()
            val_loss = 0.0
            val_start_time = time.time()
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    output = self(inputs)
                    loss = criterion(output, targets)
                    val_loss += loss

            val_loss /= len(val_loader.dataset)
            val_end_time = time.time()  # 记录验证结束时间
            epoch_end_time = time.time()
            logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Train Time: {train_end_time-train_start_time:.2f}s, "
                         f"Val Loss: {val_loss:.6f}, Val Time: {val_end_time-val_start_time:.2f}s, Total Time: {epoch_end_time-epoch_start_time:.2f}s")
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Train Time: {train_end_time-train_start_time:.2f}s, "
                  f"Val Loss: {val_loss:.6f}, Val Time: {val_end_time-val_start_time:.2f}s, Total Time: {epoch_end_time-epoch_start_time:.2f}s")
            
            #save model every 100 steps
            if (epoch+1) % 5 == 0 and save_model is True:
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
    
    def eval_latest(self, priviledge, obs_with_noise, batch_size=32, save_dir="saved_models",trajectory=128):
        model_files = [f for f in os.listdir(save_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
        
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = os.path.join(save_dir, model_files[-1])
        
        # Load the model
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        logging.info(f"Loaded model from epoch {epoch}")
        print(f"Loaded model from epoch {epoch}")
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.to(device)
        priviledge_tensor = torch.tensor(priviledge, dtype=torch.float32).to(device)
        obs_with_noise_tensor = torch.tensor(obs_with_noise, dtype=torch.float32).to(device)

        # Create dataset and data loader
        dataset = TensorDataset(obs_with_noise_tensor, priviledge_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        torch.set_printoptions(linewidth=1000)
        with torch.no_grad():
            for inputs, targets in train_loader:
                    inputs = inputs.permute(1, 0, 2)  # 调整形状为 (trajectory, batch_size, 49)
                    targets = targets.permute(1, 0, 2)
                    total_loss = 0.0
                    for t in range(trajectory):
                        input_t = inputs[t]  
                        target_t = targets[t]  
                        if input_t.shape[0]<self.memory.shape[0]:
                            input_t = torch.nn.functional.pad(input_t, 
                                        (0, 0, 0, self.memory.shape[1]-input_t.shape[0]), mode='constant', value=0)
                            target_t = torch.nn.functional.pad(target_t, 
                                        (0, 0, 0, self.memory.shape[1]-target_t.shape[0]), mode='constant', value=0)
                        #print("input_t.shape",input_t.shape)
                        if input_t.shape[1]<32:
                            break
                        output_t = self(input_t)
                        print(input_t[0],"\n",output_t[0],"\n",target_t[0])
                        time.sleep(1)
                        loss_t = criterion(output_t, target_t)
                        total_loss += loss_t
                    self.clear_memory()
                    train_loss += total_loss.item()

            train_loss /= len(train_loader.dataset)
        
    



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
        The array with added noise in different segments and noise array.
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
    noise_segments = []
    for length, noise_std in zip(segment_lengths, noise_stds):
        end_idx = start_idx + length
        segment = input_array[:, start_idx:end_idx]
        noise = np.random.randn(*segment.shape) * noise_std
        noise_segments.append(noise)
        noisy_segment = segment + noise
        noisy_segments.append(noisy_segment)
        start_idx = end_idx
    noise_array = np.concatenate(noise_segments, axis=1)
    noisy_array = np.concatenate(noisy_segments, axis=1)
    
    return noisy_array, noise_array



if __name__ == '__main__':
    model = OnlineTransformer(input_dim=45,output_dim=12, d_model=512, nhead=8, 
                                         num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)

    x = torch.randn(32, 6)  # (batch_size, input_dim)

    reconstructed = model(x)

    print("Reconstructed shape:", reconstructed.shape)  # (batch_size, input_dim)