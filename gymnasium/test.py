import gymnasium_env
import gymnasium as gym
import gymnasium_env

import warnings
import numpy as np
from PIL import Image

# 忽略 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


env = gym.make('gymnasium_env/PacmanGymEnv', speedup=5,render_mode='human')
obs, info = env.reset()
preprocessed_frame = []
output_dir = "../GIF/"
max_episode = 100
output_GIF = True

warnings.filterwarnings("ignore", category=DeprecationWarning)

env_name = "gymnasium_env/PacmanGymEnv"
env = gym.make(env_name, speedup=5.0, render_mode='human')
env = env.unwrapped #減少限制

print("environment:", env_name)
print("action space:", env.action_space.n)
# print("action:", env.unwrapped.get_action_meanings())
print("observation space:", env.observation_space.shape)

def rgb_array_loop():
    episode = 0
    for step in range(1000000):  
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        #print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Done: {done}")
        preprocessed_frame.append(obs);
        if done:
            print(f"Episode: {episode}, steps: {step}") 
            if output_GIF: 
                process_gif(episode)
                if episode == max_episode:
                    break;
                else:
                    episode += 1
            env.reset()
            

def process_gif(episode):
    frames = []
    output_path = output_dir + "episode_" + str(episode) + ".gif"
    print(f"processing {output_path}...")
    for i in preprocessed_frame:
        frames.append(Image.fromarray(i));
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=1)
    print("GIF exported!")

def human_loop():
    for step in range(1000000):  
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        #print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Episode finished!")
            print(step)
            env.reset()

if env.render_mode == "human":
    human_loop()
else:
    rgb_array_loop()

env.close()
