import numpy
from numpy._typing import NDArray
import gymnasium_env
import gymnasium
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 忽略 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


env = gymnasium.make('gymnasium_env/PacmanGymEnv', speedup=10,render_mode='human')
obs, info = env.reset()
preprocessed_frame = []
output_dir = "../GIF/"
max_episode = 100
episode = 0
output_GIF = True

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
                process_gif()
            if episode == max_episode:
                break;
            else:
                episode += 1
            env.reset()
            

def process_gif():
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
