import numpy as np
import gymnasium as gym
from PIL import Image

class PacmanEnvWrapper(gym.Wrapper):
    def __init__(self, env, k, img_size=(84,84), env_name = 'gymnasium_env/PacmanGymEnv'):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.env = env
        self.img_size = img_size
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(k, img_size[0], img_size[1]), dtype=np.float32)

    def _preprocess(self, state, th=0.182):
        # TODO(Lab-1): Image processing.
        state = np.array(Image.fromarray(state).resize(self.img_size,Image.BILINEAR))
        state = state.astype(np.float32).mean(2) / 255.
        state[state > th] = 1.0
        state[state <= th] = 0.0

        return state

    def reset(self):
        state = self.env.reset()

        # 確認是否返回了tuple，並提取圖像
        if isinstance(state, tuple):
            state = state[0]

        state = self._preprocess(state)
        state = state[np.newaxis, ...].repeat(self.k, axis=0)  # 堆疊多幀
        return state


    def step(self, action):
        state_next = []
        info =[]
        reward = 0
        terminated = False
        
        for i in range(self.k):
            if not terminated:
                state_next_f, reward_f, terminated_f, info_f = self.env.step(action)
                state_next_f = self._preprocess(state_next_f)
                reward += reward_f
                terminated = terminated_f
                info.append(info_f)
            state_next.append(state_next_f[np.newaxis, ...])
        state_next = np.concatenate(state_next, 0)
        return state_next, reward, terminated, info

