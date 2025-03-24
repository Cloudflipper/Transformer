import numpy as np
from transformer import add_segmented_noise_to_tensor
import os
from sklearn.preprocessing import StandardScaler

def normalize(arr):
    means = arr.mean(axis=(0,1), keepdims=True)
    stds = arr.std(axis=(0,1), keepdims=True)
    arr_std = (arr - means) / stds
    return arr_std

def parse_data(folder_path):
    observed_cap_posi = np.load(os.path.join(folder_path,'observed_cap_posi_data.npy'))
    real_cap_posi = np.load(os.path.join(folder_path,'cap_posi_data.npy'))
    observed_tendon_data = np.load(os.path.join(folder_path,'observed_tendon_data.npy'))
    real_tendon_data = np.load(os.path.join(folder_path,'tendon_data.npy'))
    cap_vel = np.load(os.path.join(folder_path,'cap_vel_data.npy'))
    damp_data = np.load(os.path.join(folder_path,'damp_data.npy')).reshape(-1,1)
    fric_data = np.load(os.path.join(folder_path,'friction_data.npy'))
    obs = np.hstack((observed_cap_posi,  cap_vel,observed_tendon_data))
    obs_with_noise,noise = add_segmented_noise_to_tensor(obs,[18,18,9],[0.02,1e-4,0.01])
    priviledge = np.hstack((obs_with_noise-np.hstack((real_cap_posi,  cap_vel,real_tendon_data)),damp_data,fric_data))
    return priviledge, obs_with_noise

def parse_data_for_test(root_dir="saved_data",trajectory=128):
    priviledge_list = []
    obs_with_noise_list = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            priviledge, obs_with_noise = parse_data(folder_path)
            # print("priviledge shape: ", priviledge.shape)
            # print("obs_with_noise shape: ", obs_with_noise.shape)
            priviledge = priviledge[:int(priviledge.shape[0]//trajectory)*trajectory,:].reshape(-1,trajectory,priviledge.shape[-1])
            obs_with_noise = obs_with_noise[:int(obs_with_noise.shape[0]//trajectory)*trajectory,:].reshape(-1,trajectory,obs_with_noise.shape[-1])
            priviledge_list.append(priviledge)
            obs_with_noise_list.append(obs_with_noise)
    priviledge = np.concatenate(priviledge_list, axis=0)#[:50,:]
    obs_with_noise = np.concatenate(obs_with_noise_list, axis=0)#[:50,:]
    print("priviledge shape: ", priviledge.shape)
    print("obs_with_noise shape: ", obs_with_noise.shape)
    normalize(priviledge)
    normalize(obs_with_noise)
    np.random.shuffle(priviledge)
    np.random.shuffle(obs_with_noise)
    return priviledge, obs_with_noise
            