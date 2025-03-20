from transformer import OnlineTransformer
import numpy as np
from parse import parse_data_for_test


def train():
    priviledge, obs_with_noise = parse_data_for_test()
    transformer = OnlineTransformer(input_dim=45, 
                                    output_dim=49, 
                                    d_model=128, 
                                    num_encoder_layers=3,
                                    num_decoder_layers=3,
                                    nhead=8, 
                                    dropout=0.1)
    transformer.train_all(priviledge, obs_with_noise)
    transformer.save_model('transformer_model')

if __name__ == '__main__':
    train()