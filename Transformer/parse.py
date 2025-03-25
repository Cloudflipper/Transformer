import numpy as np
from transformer import add_segmented_noise_to_tensor
import os
from sklearn.preprocessing import StandardScaler


def normalize(arr):
    means = arr.mean(axis=(0,1), keepdims=True)
    stds = arr.std(axis=(0,1), keepdims=True)
    arr_std = (arr - means) / stds
    return arr_std,means,stds

def parse_data(folder_path, omit_velocity=False):
    observed_cap_posi = np.load(os.path.join(folder_path,'observed_cap_posi_data.npy'))
    real_cap_posi = np.load(os.path.join(folder_path,'cap_posi_data.npy'))
    observed_tendon_data = np.load(os.path.join(folder_path,'observed_tendon_data.npy'))
    real_tendon_data = np.load(os.path.join(folder_path,'tendon_data.npy'))
    cap_vel = np.load(os.path.join(folder_path,'cap_vel_data.npy'))
    damp_data = np.load(os.path.join(folder_path,'damp_data.npy')).reshape(-1,1)
    fric_data = np.load(os.path.join(folder_path,'friction_data.npy'))
    
    if omit_velocity:
        obs = np.hstack((observed_cap_posi,observed_tendon_data))
        obs_with_noise,noise = add_segmented_noise_to_tensor(obs,[18,9],[0.02,0.01])
        priviledge = np.hstack((obs_with_noise-np.hstack((real_cap_posi,real_tendon_data)),cap_vel,damp_data,fric_data))
    else:
        obs = np.hstack((observed_cap_posi, cap_vel, observed_tendon_data))
        obs_with_noise,noise = add_segmented_noise_to_tensor(obs,[18,18,9],[0.02,1e-4,0.01])
        priviledge = np.hstack((obs_with_noise-np.hstack((real_cap_posi,cap_vel,real_tendon_data)),damp_data,fric_data))
    return priviledge, obs_with_noise

def parse_data_for_test(root_dir="saved_data",trajectory=128, omit_velocity=False):
    """
    Parse the data from the root_dir and return the priviledge and obs_with_noise data.
    If omit_velocity is True, the velocity data will be omitted from the obs data, by [cap_posi, tendon_data]. 
    The previlege data is [cap_posi_error, tendon_data_error, cap_vel, damp_data, fric_data].
    If omit_velocity is False, the obs data is [cap_posi, cap_vel, tendon_data].
    The previledge data is [cap_posi_error, tendon_data_error, cap_vel_error, damp_data, fric_data].
    """
    priviledge_list = []
    obs_with_noise_list = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            priviledge, obs_with_noise = parse_data(folder_path, omit_velocity)
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
    priviledge,mean,std=normalize(priviledge)
    obs_with_noise,_,_=normalize(obs_with_noise)
    np.random.shuffle(priviledge)
    np.random.shuffle(obs_with_noise)
    return priviledge, obs_with_noise,mean,std
            