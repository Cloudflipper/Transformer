from transformer import OnlineTransformer
import numpy as np
from parse import parse_data_for_test
from torch.nn import DataParallel


def train():
    priviledge, obs_with_noise,mean,std = parse_data_for_test(omit_velocity=True)
    transformer = OnlineTransformer(input_dim=27, 
                                    output_dim=49, 
                                    d_model=512, 
                                    num_encoder_layers=7,
                                    num_decoder_layers=7,
                                    nhead=8, 
                                    dropout=0.4,
                                    batch_size=256,
                                    dim_feedforward=512,
                                    mean=mean,
                                    std=std
                                    )
    transformer.train_all(priviledge, obs_with_noise,epochs=2000,batch_size=256, learning_rate=1e-2)
    transformer.save_model('transformer_model')

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
    train()