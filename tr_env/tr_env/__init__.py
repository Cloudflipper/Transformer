from gym.envs.registration import register

register(
    id='tr_env-v0',
    entry_point='tr_env.tr_env.envs:tr_env',
    max_episode_steps=5000,
)