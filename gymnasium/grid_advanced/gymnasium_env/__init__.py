from gymnasium.envs.registration import register

register(
    id="gymnasium_env/PacmanGymEnv",
    entry_point="gymnasium_env.envs:PacmanGymEnv",
)
