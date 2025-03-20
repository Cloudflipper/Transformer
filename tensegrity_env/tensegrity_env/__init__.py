from gym.envs.registration import register

register(
    id='tensegrity_env-v0',
    entry_point='tensegrity_env.envs:tensegrity_env',
    max_episode_steps=5000,
)