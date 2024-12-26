import numpy as np
import gymnasium as gym
from PIL import Image

class PacmanEnvWrapper(gym.Wrapper):
    def __init__(self, env, k, env_name = 'gymnasium_env/PacmanGymEnv'):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.env = env
        self.obs_shape = self.env.observation_space.shape
        

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(k, *self.obs_shape), dtype=np.uint8)


    def reset(self, seed=1):
        observation = self.env.reset()

        # 確認是否返回了tuple，並提取圖像
        if isinstance(observation, tuple):
            observation = observation[0]

        stacked = []
        for _ in range(self.k):
            stacked.append(observation)

        observation = np.concatenate(stacked, axis=0)
        return observation


    def step(self, action):
        state_next = []
        info = []
        reward = 0
        terminated = False
        truncated = False
        
        for _ in range(self.k):
            if not terminated:
                state_next_f, reward_f, terminated_f, truncated_f, info_f = self.env.step(action)
                reward += reward_f
                terminated = terminated_f
                truncated = truncated_f
                info.append(info_f)
            else:
                state_next_f = np.zeros((1, self.obs_shape[1], self.obs_shape[2]))

            state_next.append(state_next_f)

        state_next = np.concatenate(state_next, axis=0)
        return state_next, reward, terminated, truncated, info

