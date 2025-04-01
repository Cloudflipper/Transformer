from transformer import OnlineTransformer
import numpy as np
from parse import parse_data_for_test
from torch.nn import DataParallel
import torch
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):

    def __init__(self, num_samples=1024):

        self.num_samples = num_samples

        self.noisy_data = torch.randn(num_samples, 27)

        self.priv_data = torch.randn(num_samples, 49)

    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.noisy_data[idx], self.priv_data[idx]


def train():
    priviledge, obs_with_noise,mean_p,std_p,mean_o,std_o = parse_data_for_test(omit_velocity=True)
    transformer = OnlineTransformer(input_dim=27, 
                                    output_dim=49, 
                                    d_model=512, 
                                    num_encoder_layers=7,
                                    num_decoder_layers=7,
                                    nhead=8, 
                                    dropout=0.4,
                                    batch_size=256,
                                    dim_feedforward=512,
                                    )
    transformer.train_all(priviledge, obs_with_noise,epochs=2000,batch_size=256, learning_rate=1e-2)
    transformer.save_model('transformer_model')

def test():
    model = OnlineTransformer(input_dim=27, 
                                    output_dim=49, 
                                    d_model=512, 
                                    num_encoder_layers=7,
                                    num_decoder_layers=7,
                                    nhead=8, 
                                    dropout=0.4,
                                    batch_size=256,
                                    dim_feedforward=512,
                                    )
    dataset = CustomDataset(num_samples=1024)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    for epoch in range(100):
        for batch in dataloader:
            noisy, priv = batch
            
            # Single update step
            recon, loss = model.update(
                noised_input=noisy,
                previledge=priv,
                epoch=epoch,
                learning_rate=1e-4
            )
            
            # Get enhanced features
            features = model.get_feature(type_index=1)
            
            print(f"Epoch {epoch} Loss: {loss:.4f}")
            print(f"Feature shape: {features.shape}")

def eval():
    priviledge, obs_with_noise, mean, std = parse_data_for_test()
    transformer = OnlineTransformer(input_dim=45, 
                                    output_dim=49, 
                                    d_model=256, 
                                    num_encoder_layers=5,
                                    num_decoder_layers=5,
                                    nhead=8, 
                                    dropout=0.1,
                                    batch_size=48,
                                    )
    transformer.eval_latest(priviledge,obs_with_noise)


if __name__ == '__main__':
    test()