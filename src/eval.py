import gymnasium
import gymnasium_env

import util
import dqn
import env_wrapper
import torch
import os

env_name = 'gymnasium_env/PacmanGymEnv'
env = gymnasium.make(env_name, speedup = 1.0, render_mode='human');
env_pacman = env_wrapper.PacmanEnvWrapper(env, k=4, img_size=(84,84))

stack_frames = 4
img_size = (84, 84)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_variant = "Double"

if model_variant == "DoubleDueling":
    agent = dqn.DeepQNetwork(
        n_actions = env.action_space.n,
        input_shape = [stack_frames, *img_size],
        qnet = dqn.QNet,
        device = device,
        learning_rate = 2e-4,
        reward_decay = 0.9,
        replace_target_iter = 1000,
        memory_size = 100000,
        batch_size = 32,)
elif model_variant == "Double":
    agent = dqn.DeepQNetwork(
        n_actions = env.action_space.n,
        input_shape = [stack_frames, *img_size],
        qnet = dqn.QNet,
        device = device,
        learning_rate = 2e-4,
        reward_decay = 0.9,
        replace_target_iter = 1000,
        memory_size = 100000,
        batch_size = 32,)
else:
    print(f'Invalid Model Name: "{model_variant}"')
    print('Valid ones: "Double", "DoubleDueling"')
    exit()

save_dir = os.path.join(os.getcwd(), '..', 'save');
gif_dif = os.path.join(os.getcwd(), '..', 'GIF');

agent.save_load_model(op="load", path=save_dir, fname="qnet.pt")
env_pacman = env_wrapper.PacmanEnvWrapper(env, k=4, img_size=(84,84))
img_buffer = util.play(env_pacman, agent, stack_frames, img_size, 0.2)
util.save_gif(img_buffer, "eval.gif")
