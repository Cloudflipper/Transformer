from transformer import OnlineTransformer
import numpy as np
from parse import parse_data_for_test
from torch.nn import DataParallel


def train():
    priviledge, obs_with_noise = parse_data_for_test()
    transformer = OnlineTransformer(input_dim=45, 
                                    output_dim=49, 
                                    d_model=1024, 
                                    num_encoder_layers=15,
                                    num_decoder_layers=15,
                                    nhead=16, 
                                    dropout=0.3,
                                    batch_size=64,
                                    dim_feedforward=1024)
    transformer.train_all(priviledge, obs_with_noise,epochs=2000,batch_size=64, learning_rate=1e-3,
                                    )
    transformer.save_model('transformer_model')

def eval():
    priviledge, obs_with_noise = parse_data_for_test()
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
    train()