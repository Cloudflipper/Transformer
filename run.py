import gym
import numpy as np
from stable_baselines3 import SAC, TD3, A2C, PPO
import mujoco
import os
import argparse
# import tensegrity_env
import tr_env.tr_env.envs.tr_env
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

#print(gym.envs.registry.keys())
env = gym.make('tr_env-v0', render_mode="human")

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if "reward_forward" in self.locals:
            reward = self.locals["reward_forward"]
            self.logger.record("reward_forward", reward)
        if "reward_ctrl" in self.locals:
            reward = self.locals["reward_ctrl"]
            self.logger.record("reward_ctrl", reward)
        return True


def train(env, sb3_algo, log_dir, model_dir, delay, lr_SAC, gpu_idx, starting_point = None):
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    chosen_device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")

    if sb3_algo == 'SAC':
        if delay == 10:
            # take 10 steps in the environment, then update critic 10 times,
            # updating the actor every 2nd time (so 5 times total)
            if starting_point is None:
                model = SAC('MlpPolicy', env, learning_rate=lr_SAC, verbose=1, device=chosen_device, tensorboard_log=log_dir, 
                        train_freq=10, gradient_steps=10, target_update_interval=2)
            else:
                model = SAC.load(starting_point, env, learning_rate=lr_SAC, verbose=1, device=chosen_device, tensorboard_log=log_dir,
                        train_freq=10, gradient_steps=10, target_update_interval=2)
            
        elif delay == 100:
            # take 100 steps in the environment, then update critic 100 times, 
            # updating the actor every 10th time (so 10 times total)
            if starting_point is None:
                model = SAC('MlpPolicy', env, learning_rate=lr_SAC, verbose=1, device=chosen_device, tensorboard_log=log_dir, 
                        train_freq=100, gradient_steps=100, target_update_interval=10)
            else:
                model = SAC.load(starting_point, env, learning_rate=lr_SAC, verbose=1, device=chosen_device, tensorboard_log=log_dir, 
                        train_freq=100, gradient_steps=100, target_update_interval=10)
            
        
        else:
            # take 1 step in the environment, then update critic 1 time, then update actor 1 time
            if starting_point is None:
                model = SAC('MlpPolicy', env, learning_rate=lr_SAC, verbose=1, device=chosen_device, tensorboard_log=log_dir)
            else:
                model = SAC.load(starting_point, env, learning_rate=lr_SAC, verbose=1, device=chosen_device, tensorboard_log=log_dir)
            
            

    elif sb3_algo == 'TD3':
        # default learning rate: 0.001
        if starting_point is None:
            model = TD3('MlpPolicy', env, verbose=1, device=chosen_device, tensorboard_log=log_dir)
        else:
            model = TD3.load(starting_point, env, verbose=1, device=chosen_device, tensorboard_log=log_dir)


    elif sb3_algo == 'A2C':
        if starting_point is None:
            model = A2C('MlpPolicy', env, verbose=1, device=chosen_device, tensorboard_log=log_dir)
        else:
            model = A2C.load(starting_point, env, verbose=1, device=chosen_device, tensorboard_log=log_dir)

    elif sb3_algo == 'PPO':
        if starting_point is None:
            model = PPO('MlpPolicy', env, verbose=1, device=chosen_device, tensorboard_log=log_dir)
        else:
            model = PPO.load(starting_point, env, verbose=1, device=chosen_device, tensorboard_log=log_dir)
    else:
        print('Algorithm not found')
        return

    print(f"Using {torch.cuda.device_count()} GPUs!")

    TIMESTEPS = 25000
    iters = 0

    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")
        print(f"step: {TIMESTEPS*iters}")

def test(env, sb3_algo, path_to_model, saved_data_dir, simulation_seconds):

    if sb3_algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    elif sb3_algo == 'PPO':
        model = PPO.load(path_to_model, env=env)
    else:
        print('Algorithm not found')
        return
    damping_list = ["100","200","300","400","500","700","1000","2000","3000","5000"]
    friction_list = ["1.0 0.01 0.001","0.8 0.005 0.001","0.5 0.005 0.001","0.3 0.005 0.001","0.2 0.005 0.001","0.15 0.005 0.001","0.1 0.005 0.001"]
    for damping in tqdm(damping_list, desc="damping"):
        
        
        for tendon in ["td6","td7","td8"]:
            tendon_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_TENDON, tendon)
            env.unwrapped.model.tendon_damping[tendon_id] = float(damping)
            print(f"Modified damping for tendon to {damping}")
            
            
        for friction in tqdm(friction_list, desc="friction", leave=False):
            new_f = np.array(friction.split()).astype(float)
            for i in range(env.unwrapped.model.ngeom):
                env.unwrapped.model.geom_friction[i,:] = new_f
            print(f"Modified friction for geom to {new_f}")
            
            
            data_pos = saved_data_dir+"/saved_data "+damping+" "+friction
            os.makedirs(data_pos, exist_ok=True)

            obs = env.reset()[0]
            done = False
            extra_steps = 500
            
            dt = env.dt
            actions_list = []
            tendon_length_list = []
            observed_tendon_length_list = []
            cap_posi_list = []
            observed_cap_posi_list = []
            total_bar_contact_list = []
            reward_forward_list = []
            reward_ctrl_list = []
            waypt_list = []
            x_pos_list = []
            y_pos_list = []
            cap_vel_list = []
            damp_note = []
            friction_note = []
            
            iter = int(simulation_seconds/dt)
            for i in range(iter):
                action, _ = model.predict(obs)
                obs, _, done, _, info = env.step(action)

                actions_list.append(action)
                #the tendon lengths are the last 9 observations
                # tendon_length_list.append(obs[-9:])
                tendon_length_list.append(info["tendon_length"])
                observed_tendon_length_list.append(obs[-9:])
                cap_posi_list.append(info["real_observation"][:18])
                observed_cap_posi_list.append(obs[:18])
                reward_forward_list.append(info["reward_forward"])
                reward_ctrl_list.append(info["reward_ctrl"])
                waypt_list.append(info["waypt"])
                x_pos_list.append(info["x_position"])
                y_pos_list.append(info["y_position"])
                cap_vel_list.append(info["cap_vel"])
                damp_note.append(float(damping))
                friction_note.append(np.array(friction.split()).astype(float))
                total_bar_contact = 0
                for j,contact in enumerate(env.data.contact):
                    if contact.geom1 != 0 and contact.geom2 != 0: # neither geom is 0, which is ground. so contact is between bars
                        forcetorque = np.zeros(6)
                        mujoco.mj_contactForce(env.model, env.data, j, forcetorque)
                        force_mag = np.sqrt(forcetorque[0]**2 + forcetorque[1]**2 + forcetorque[2]**2)
                        total_bar_contact += force_mag
                total_bar_contact_list.append(total_bar_contact)

                if done:
                    extra_steps -= 1

                    if extra_steps < 0:
                        break

            action_array = np.array(actions_list)
            tendon_length_array = np.array(tendon_length_list)
            observed_tendon_length_array = np.array(observed_tendon_length_list)
            cap_posi_array = np.array(cap_posi_list)
            observed_cap_posi_array = np.array(observed_cap_posi_list)
            total_bar_contact_array = np.array(total_bar_contact_list)
            reward_forward_array = np.array(reward_forward_list)
            reward_ctrl_array = np.array(reward_ctrl_list)
            waypt_array = np.array(waypt_list)
            x_pos_array = np.array(x_pos_list)
            y_pos_array = np.array(y_pos_list)
            cap_vel_array = np.array(cap_vel_list)
            np.save(os.path.join(data_pos, "action_data.npy"),action_array)
            np.save(os.path.join(data_pos, "tendon_data.npy"),tendon_length_array)
            np.save(os.path.join(data_pos, "observed_tendon_data.npy"),observed_tendon_length_array)
            np.save(os.path.join(data_pos, "cap_posi_data.npy"),cap_posi_array)
            np.save(os.path.join(data_pos, "observed_cap_posi_data.npy"),observed_cap_posi_array)
            np.save(os.path.join(data_pos, "total_bar_contact_data.npy"),total_bar_contact_array)
            np.save(os.path.join(data_pos, "reward_forward_data.npy"),reward_forward_array)
            np.save(os.path.join(data_pos, "reward_ctrl_data.npy"),reward_ctrl_array)
            np.save(os.path.join(data_pos, "waypt_data.npy"),waypt_array)
            np.save(os.path.join(data_pos, "x_pos_data.npy"),x_pos_array)
            np.save(os.path.join(data_pos, "y_pos_data.npy"),y_pos_array)
            np.save(os.path.join(data_pos, "cap_vel_data.npy"),cap_vel_array)
            np.save(os.path.join(data_pos, "damp_data.npy"),np.array(damp_note))
            np.save(os.path.join(data_pos, "friction_data.npy"),np.array(friction_note))

def test3(env, sb3_algo, path_to_model_tracking, path_to_model_ccw, path_to_model_cw, saved_data_dir, simulation_seconds):
    if sb3_algo == 'SAC':
        model_tracking = SAC.load(path_to_model_tracking, env=env)
        model_ccw = SAC.load(path_to_model_ccw, env=env)
        model_cw = SAC.load(path_to_model_cw, env=env)
    elif sb3_algo == 'TD3':
        model_tracking = TD3.load(path_to_model_tracking, env=env)
        model_ccw = TD3.load(path_to_model_ccw, env=env)
        model_cw = TD3.load(path_to_model_cw, env=env)
    elif sb3_algo == 'A2C':
        model_tracking = A2C.load(path_to_model_tracking, env=env)
        model_ccw = A2C.load(path_to_model_ccw, env=env)
        model_cw = A2C.load(path_to_model_cw, env=env)
    elif sb3_algo == 'PPO':
        model_tracking = PPO.load(path_to_model_tracking, env=env)
        model_ccw = PPO.load(path_to_model_ccw, env=env)
        model_cw = PPO.load(path_to_model_cw, env=env)
    else:
        print('Algorithm not found')
        return
    
    obs = env.reset()[0]
    done = False
    extra_steps = 500
    waypt_threshold = 0.2

    dt = env.dt
    waypt_list = np.array([[1,1],
                           [2,0]])
    waypt_list = np.array([[0,2],
                           [2,0],
                           [4,2],
                           [4,0]])
    x_pos_list = []
    y_pos_list = []
    del_yaw_list = []
    iter = int(simulation_seconds/dt)

    tendon_loop_init = obs[36:42]

    counter = 0
    for idx_wp in range(waypt_list.shape[0]):
        switch_waypt = False
        turn_state_open = True
        obs, _, done, _, info = env.step(tendon_loop_init)
        while switch_waypt==False and counter < iter and extra_steps >= 0:
            pos_rel_s0 = obs[0:3]
            pos_rel_s1 = obs[3:6]
            pos_rel_s2 = obs[6:9]
            pos_rel_s3 = obs[9:12]
            pos_rel_s4 = obs[12:15]
            pos_rel_s5 = obs[15:18]
            pos_rbt = -obs[45:47]
            tracking_vec = np.array([waypt_list[idx_wp][0] - pos_rbt[0], waypt_list[idx_wp][1] - pos_rbt[1]])
            tgt_yaw = np.arctan2(tracking_vec[1], tracking_vec[0])
            left_com = (pos_rel_s0 + pos_rel_s2 + pos_rel_s4)/3
            right_com = (pos_rel_s1 + pos_rel_s3 + pos_rel_s5)/3
            rbt_yaw = np.arctan2(right_com[0] - left_com[0], left_com[1] - right_com[1])
            del_yaw = tgt_yaw - rbt_yaw
            if del_yaw > np.pi:
                del_yaw -= 2*np.pi
            elif del_yaw <= -np.pi:
                del_yaw += 2*np.pi
            del_yaw_list.append(del_yaw)
            
            if del_yaw > np.pi/15 and turn_state_open:
                obs_turn = obs
                obs_turn[45] = 0
                obs_turn[46] = 0
                obs_turn[47] = 0
                action, _ = model_ccw.predict(obs_turn)
            elif del_yaw < 0 and turn_state_open:
                obs_turn = obs
                obs_turn[45] = 0
                obs_turn[46] = 0
                obs_turn[47] = 0
                action, _ = model_cw.predict(obs_turn)
            else:
                obs_tracking = obs
                tracking_drct = tracking_vec/np.linalg.norm(tracking_vec)
                obs_tracking[45] = tracking_drct[0]
                obs_tracking[46] = tracking_drct[1]
                obs_tracking[47] = tgt_yaw
                action, _ = model_tracking.predict(obs_tracking)
                turn_state_open = False

            # ang_bia = 0 * np.pi / 180  
            # pos_tgt_rel = np.array([np.cos(rbt_yaw+ang_bia), np.sin(rbt_yaw+ang_bia)])
            # obs_tracking = obs
            # obs_tracking[45] = pos_tgt_rel[0]
            # obs_tracking[46] = pos_tgt_rel[1]
            # obs_tracking[47] = rbt_yaw+ang_bia
            # action, _ = model_tracking.predict(obs_tracking)

            obs, _, done, _, info = env.step(action)

            # waypt_list.append(info["waypt"])
            x_pos_list.append(info["x_position"])
            y_pos_list.append(info["y_position"])

            if np.linalg.norm(np.array([info["x_position"], info["y_position"]]) - waypt_list[idx_wp]) < waypt_threshold:
                switch_waypt = True
                tendon_loop_init = obs[36:42]
                print(f"waypt {idx_wp} reached")

            counter += 1

            if done:
                extra_steps -= 1

                if extra_steps < 0:
                    break
    
    np.save(os.path.join(saved_data_dir, "waypt_data.npy"),np.array(waypt_list))
    np.save(os.path.join(saved_data_dir, "x_pos_data.npy"),np.array(x_pos_list))
    np.save(os.path.join(saved_data_dir, "y_pos_data.npy"),np.array(y_pos_list))
    np.save(os.path.join(saved_data_dir, "del_yaw_data.npy"),np.array(del_yaw_list))

    return

def tracking_test(env, sb3_algo, path_to_model, saved_data_dir, simulation_seconds, episode_num):

    if sb3_algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    elif sb3_algo == 'PPO':
        model = PPO.load(path_to_model, env=env)
    else:
        print('Algorithm not found')
        return
    
    os.makedirs(saved_data_dir, exist_ok=True)

    oripoint_list = []
    waypt_list = []
    xy_pos_list = []
    for i in range(episode_num):
        obs = env.reset()[0]
        done = False
        extra_steps = 500

        dt = env.dt
        iter = int(simulation_seconds/dt)
        for j in range(iter):
            action, _ = model.predict(obs)
            obs, _, done, _, info = env.step(action)

            if done:
                extra_steps -= 1

                if extra_steps < 0:
                    break
        oripoint_list.append(info["oripoint"])
        waypt_list.append(info["waypt"])
        xy_pos_list.append(np.array([info["x_position"], info["y_position"]]))
    oripoint_array = np.array(oripoint_list)
    waypt_array = np.array(waypt_list)
    xy_pos_array = np.array(xy_pos_list)

    waypt_array = waypt_array - oripoint_array
    xy_pos_array = xy_pos_array - oripoint_array
    oripoint_array = oripoint_array - oripoint_array
    for i in range(episode_num):
        waypt_ang = np.arctan2(waypt_array[i,1], waypt_array[i,0])
        rot_mat = np.array([[np.cos(waypt_ang), np.sin(waypt_ang)], [np.sin(waypt_ang), -np.cos(waypt_ang)]])
        waypt_array[i] = np.dot(rot_mat, waypt_array[i])
        xy_pos_array[i] = np.dot(rot_mat, xy_pos_array[i])

    np.save(os.path.join(saved_data_dir, "waypt_data.npy"),waypt_array)
    np.save(os.path.join(saved_data_dir, "xy_pos_data.npy"),xy_pos_array)
    np.save(os.path.join(saved_data_dir, "oripoint_data.npy"),oripoint_array)


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', metavar='path_to_model')
    parser.add_argument('--test3', metavar='path_to_model', nargs=3)
    parser.add_argument('--tracking_test', metavar='path_to_model')
    parser.add_argument('--starting_point', metavar='path_to_starting_model')
    parser.add_argument('--env_xml', default="3prism_jonathan_steady_side.xml", type=str,
                        help="ther name of the xml file for the mujoco environment, should be in same directory as run.py")
    parser.add_argument('--sb3_algo', default="SAC", type=str, choices=["SAC", "TD3", "A2C", "PPO"],
                        help='StableBaseline3 RL algorithm: SAC, TD3, A2C, PPO')
    parser.add_argument('--desired_action', default="straight", type=str, choices=["straight", "turn", "tracking", "aiming", "vel_track"],
                        help="either straight or turn, determines what the agent is learning")
    parser.add_argument('--desired_direction', default=1, type=int, choices=[-1, 1], 
                        help="either 1 or -1, 1 means roll forward or turn counterclockwise,-1 means roll backward or turn clockwise")
    parser.add_argument('--delay', default="1", type=int, choices=[1, 10, 100],
                        help="how many steps to take in environment before updating critic\
                        Can be 1, 10, or 100. Default is 1, which worked best when training")
    parser.add_argument('--terminate_when_unhealthy', default="yes", type=str,choices=["yes", "no"],
                         help="Determines if the training is reset when the tensegrity stops moving or not, default is True.\
                            Best results are to set yes when training to move straight and set no when training to turn")
    parser.add_argument('--contact_with_self_penatly', default= 0.0, type=float,
                        help="The penalty multiplied by the total contact between bars, which is then subtracted from the reward.\
                        By default this is 0.0, meaning there is no penalty for contact.")
    parser.add_argument('--log_dir', default="logs", type=str,
                        help="The directory where the training logs will be saved")
    parser.add_argument('--model_dir', default="models", type=str,
                        help="The directory where the trained models will be saved")
    parser.add_argument('--saved_data_dir', default="saved_data", type=str)
    parser.add_argument('--simulation_seconds', default=30, type=int,
                         help="time in seconds to run simulation when testing, default is 30 seconds")
    parser.add_argument('--lr_SAC', default=3e-4, type=float,
                        help="learning rate for SAC, default is 3e-4")
    parser.add_argument('--gpu_idx', default=2, type=int,
                        help="index of the GPU to use, default is 2")
    args = parser.parse_args()

    if args.terminate_when_unhealthy == "no":
        terminate_when_unhealthy = False
    else:
        terminate_when_unhealthy = True

    if args.train:
        gymenv = gym.make("tr_env-v0", render_mode="None",
                          xml_file=os.path.join(os.getcwd(),args.env_xml),
                          is_test = False,
                          desired_action = args.desired_action,
                          desired_direction = args.desired_direction,
                          terminate_when_unhealthy = terminate_when_unhealthy)
        if args.starting_point and os.path.isfile(args.starting_point):
            train(gymenv, args.sb3_algo, args.log_dir, args.model_dir, args.delay, lr_SAC=args.lr_SAC, gpu_idx=args.gpu_idx, starting_point= args.starting_point)
        else:
            train(gymenv, args.sb3_algo, args.log_dir, args.model_dir, args.delay, lr_SAC=args.lr_SAC, gpu_idx=args.gpu_idx)

    if(args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make("tr_env-v0", render_mode='human',
                            xml_file=os.path.join(os.getcwd(),args.env_xml),
                            is_test = True,
                            desired_action = args.desired_action,
                            desired_direction = args.desired_direction)
            test(gymenv, args.sb3_algo, path_to_model=args.test, saved_data_dir=args.saved_data_dir, simulation_seconds = args.simulation_seconds)
        else:
            print(f'{args.test} not found.')

    if(args.test3):
        if os.path.isfile(args.test3[0]) and os.path.isfile(args.test3[1]) and os.path.isfile(args.test3[2]):
            gymenv = gym.make("tr_env-v0", render_mode='human',
                            xml_file=os.path.join(os.getcwd(),args.env_xml),
                            is_test = True,
                            desired_action = args.desired_action,
                            desired_direction = args.desired_direction)
            test3(gymenv, args.sb3_algo, path_to_model_tracking=args.test3[0], path_to_model_ccw=args.test3[1], path_to_model_cw=args.test3[2], saved_data_dir=args.saved_data_dir, simulation_seconds = args.simulation_seconds)
        else:
            print(f'{args.test3} not found.')
    
    if(args.tracking_test):
        if os.path.isfile(args.tracking_test):
            gymenv = gym.make("tr_env-v0", render_mode='None',
                            xml_file=os.path.join(os.getcwd(),args.env_xml),
                            is_test = True,
                            desired_action = "tracking",
                            desired_direction = args.desired_direction)
            tracking_test(gymenv, args.sb3_algo, path_to_model=args.tracking_test, saved_data_dir=args.saved_data_dir, simulation_seconds = args.simulation_seconds, episode_num = 100)
        else:
            print(f'{args.tracking_test} not found.')


