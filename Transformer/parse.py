import numpy as np
from transformer import add_segmented_noise_to_tensor
import os

def parse_data(folder_path):
    observed_cap_posi = np.load(os.path.join(folder_path,'observed_cap_posi_data.npy'))
    real_cap_posi = np.load(os.path.join(folder_path,'cap_posi_data.npy'))
    observed_tendon_data = np.load(os.path.join(folder_path,'observed_tendon_data.npy'))
    real_tendon_data = np.load(os.path.join(folder_path,'tendon_data.npy'))
    cap_vel = np.load(os.path.join(folder_path,'cap_vel_data.npy'))
    damp_data = np.load(os.path.join(folder_path,'damp_data.npy')).reshape(-1,1)
    fric_data = np.load(os.path.join(folder_path,'friction_data.npy'))
    obs = np.hstack((observed_cap_posi,  cap_vel,observed_tendon_data))
    obs_with_noise = add_segmented_noise_to_tensor(obs,[18,18,9],[0.05,1e-4,0.05])
    priviledge = np.hstack((real_cap_posi,  cap_vel,real_tendon_data,damp_data,fric_data))
    return priviledge, obs_with_noise

def parse_data_for_test(root_dir="saved_data"):
    priviledge_list = []
    obs_with_noise_list = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            priviledge, obs_with_noise = parse_data(folder_path)
            priviledge_list.append(priviledge)
            obs_with_noise_list.append(obs_with_noise)
    priviledge = np.concatenate(priviledge_list, axis=0)
    obs_with_noise = np.concatenate(obs_with_noise_list, axis=0)
    print("priviledge shape: ", priviledge.shape)
    print("obs_with_noise shape: ", obs_with_noise.shape)
    np.random.shuffle(priviledge)
    np.random.shuffle(obs_with_noise)
    priviledge = priviledge/(np.max(priviledge, axis=0) - np.min(priviledge, axis=0)) 
    obs_with_noise = obs_with_noise/(np.max(obs_with_noise, axis=0) - np.min(obs_with_noise, axis=0)) 
    return priviledge, obs_with_noise
            