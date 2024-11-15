import os
import numpy as np
from PIL import Image

project_root = os.getcwd()

def save_gif(img_buffer, fname, gif_path=os.path.join(os.getcwd(), '..', 'GIF')):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    img_buffer[0].save(os.path.join(gif_path, fname), save_all=True, append_images=img_buffer[1:], duration=3, loop=0)

def play(env, agent, stack_frames, img_size, randomized_ratio):
    # Reset environment.
    state = env.reset()
    img_buffer = [Image.fromarray(state[0]*255)]

    # Initialize information.
    step = 0
    total_reward = 0

    # One episode.
    while True:
        # Select action.
        action = agent.choose_action(state, randomized_ratio)

        # Get next stacked state.
        state_next, reward, done, info = env.step(action)
        if step % 2 == 0:
            img_buffer.append(Image.fromarray(state_next[0]*255))

        state = state_next.copy()
        step += 1
        total_reward += reward
        print('\rStep: {:3d} | Reward: {:.3f} / {:.3f} | Action: {:.3f} | Info: {}'.format(step, reward, total_reward, action, info[0]), end="")

        if done or step > 2000:
            print()
            break

    return img_buffer

def epsilon_compute(frame_id, epsilon_max=1, epsilon_min=0.05, epsilon_decay=100000):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-frame_id / epsilon_decay)

def train(env, agent, stack_frames, img_size, save_path="save", max_steps=1000000):
    total_step = 0
    episode = 0
    
    # 嘗試加載模型和訓練狀態
    try:
        print("Loading model and training status...")
        status = agent.save_load_model(op="load", path=save_path, fname="qnet.pt")
        total_step = status["learn_step_counter"]
        episode = status["memory_counter"]
        print(f"Resuming training from total_step={total_step}, episode={episode}")
    except FileNotFoundError:
        print("No previous model found. Starting training from scratch.")
    except KeyError as e:
        print(f"Missing key in checkpoint: {e}")

    while total_step <= max_steps:
        # Reset environment.
        state = env.reset()

        # 如果 state 是 tuple，提取圖像
        if isinstance(state, tuple):
            state = state[0]

        # Initialize information.
        step = 0
        total_reward = 0
        loss = 0

        # One episode.
        while True:
            # TODO(Lab-6): Select action.
            epsilon = epsilon_compute(total_step)
            action = agent.choose_action(state, epsilon)

            # Get next observation.
            obs, reward, terminated, info = env.step(action)

            # 如果 obs 是 tuple，提取圖像
            if isinstance(obs, tuple):
                obs = obs[0]

            # 判斷是否遊戲結束
            done = terminated

            # TODO(Lab-7): Train RL model.
            # Store transition and learn.
            agent.store_transition(state, action, reward, obs, done)
            if total_step > 4 * agent.batch_size:
                loss = agent.learn()

            state = obs.copy()  # 更新狀態
            step += 1
            total_step += 1
            total_reward += reward

            # 確保 loss 為浮點數以便於打印
            if total_step % 8 == 0 or done:
                print('\rEpisode: {:3d} | Step: {:3d} / {:3d} | Reward: {:.3f} / {:.3f} | Loss: {:.3f} | Epsilon: {:.3f}'\
                    .format(agent.episode, step, total_step, reward, total_reward, loss, epsilon), end="")

            if total_step % 4000 == 0:
                print("\nSave Model ...")
                agent.save_load_model(
                    op="save",
                    path=save_path,
                    fname="qnet.pt"
                )
                print("Generate GIF ...")
                img_buffer = play(env, agent, stack_frames, img_size, 0.2)
                save_gif(img_buffer, "train_" + str(total_step).zfill(6) + ".gif")
                print("Done !!")

            if done or step > 2000:
                agent.episode += 1
                print()
                break

        if total_step > max_steps:
            break

