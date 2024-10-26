import numpy
import gymnasium_env
import gymnasium
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 忽略 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

colab = True

env = gymnasium.make('gymnasium_env/PacmanGymEnv', speedup=5.0,render_mode='human')
obs, info = env.reset()

root = tk.Tk()
root.title("Tkinter Window with Matplotlib Image")
root.geometry("800x600")

def rgb_array_loop():
    for step in range(1000000):  
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        #print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Episode finished!")
            print(step)
            env.reset()

def rgb_array_loop_colab():
    for step in range(1000000):  
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        a = env.render()
        a = Image.fromarray(a)
        plt.imshow(a)
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.001)
        plt.clf()
        #print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Episode finished!")
            print(step)
            env.reset()


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

if env.render_mode == 'human':
    human_loop()
elif colab == True:
    rgb_array_loop_colab()
else:
    rgb_array_loop()
env.close()
