import os
import numpy as np
import pandas as pd
import shutil
from PIL import Image

project_root = os.getcwd()

def save_gif(img_buffer, fname, gif_path=os.path.join(os.getcwd(), '..', 'GIF')):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    img_buffer[0].save(os.path.join(gif_path, fname), save_all=True, append_images=img_buffer[1:], duration=3, loop=0)

def play(env, agent, stack_frames, img_size, randomized_ratio):
    # Reset environment.
    state = env.reset()
    img_buffer = [Image.fromarray(state[0])]

    # Initialize information.
    step = 0
    total_reward = 0

    # One episode.
    while True:
        # Select action.
        action, q = agent.choose_action(state, randomized_ratio)
        # Get next stacked state.
        state_next, reward, terminated, truncated, info = env.step(action)
        if step % 2 == 0:
            img_buffer.append(Image.fromarray(state_next[0]))

        state = state_next.copy()
        step += 1
        total_reward += reward
        print('\rStep: {:3d} | Reward: {:.3f} / {:.3f} | Action: {:.3f} | Info: {}'.format(step, reward, total_reward, action, info[0]), end="")

        if terminated or step > 2000:
            print()
            break

    return img_buffer

def epsilon_compute(frame_id, epsilon_max=1, epsilon_min=0.05, epsilon_decay=200000):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-frame_id / epsilon_decay)

def write_to_csv(episode: int, total_steps: int, loss_arr: list, total_reward: int, episode_score: int, csv_filepath: str, epsilon: float):

    # Calculate average of loss function value for each episode
    avg_loss = np.sum(loss_arr) / len(loss_arr)
    print(type(avg_loss))

    # Store data into csv
    df = pd.DataFrame([{
        "Episodes": episode,
        "Total_Steps": total_steps,
        "Loss": avg_loss,
        "Total_Reward": total_reward,
        "Score": episode_score,
        "Epsilon": epsilon,
    }])

    write_header = not os.path.exists(csv_filepath)
    df.to_csv(csv_filepath, mode='a', header=write_header, index=False)
    print(f"Metrics of episode {episode} appended to {csv_filepath}!")

    return


def train(env, agent, stack_frames, img_size, save_path="save", max_steps=1000000, session_name="default", max_episodes=10000):
    total_step = 0
    episode = 0

    # 初始化紀錄損失值與步數
    loss_values = []


    # 確保保存路徑存在
    os.makedirs(save_path, exist_ok=True)
    csv_filename = f"training_metrics_{session_name}.csv"
    csv_path = os.path.join(save_path, csv_filename)
    model_filename = f"qnet_{session_name}.pt"

    # 嘗試加載模型和訓練狀態
    try:
        print("Loading model and training status...")
        status = agent.save_load_model(op="load", path=save_path, fname=model_filename)
        total_step = status["learn_step_counter"]
        episode = status["episode"]
        print(f"Resuming training from total_step={total_step}, episode={episode}")
    except FileNotFoundError:
        print("No previous model found. Starting training from scratch.")
    except KeyError as e:
        print(f"Missing key in checkpoint: {e}")


    while total_step <= max_steps:
        # Reset environment.
        state = env.reset()

        # Initialize information.
        step = 0
        total_reward = 0
        loss = 0

        # Make sure agent episode does not exceed the limit
        if agent.episode > max_episodes:
            break

        # One episode.
        while True:
            loss_values = []
            # TODO(Lab-6): Select action.
            epsilon = epsilon_compute(total_step)
            action, q = agent.choose_action(state, epsilon)

            # Get next observation.
            obs, reward, terminated, truncated, info = env.step(action)

            # 如果 obs 是 tuple，提取圖像
            if isinstance(obs, tuple):
                obs = obs[0]

            # 判斷是否遊戲結束
            done = terminated

            # Store transition and learn.
            agent.store_transition(state, action, reward, obs, done)

            if total_step > 4 * agent.batch_size:
                loss = agent.learn()

            state = obs.copy()  # 更新狀態
            step += 1
            total_step += 1
            total_reward += reward


            # Print status 
            if total_step % 10 == 0 or done:
                print('\rEpisode: {:3d} | Step: {:3d} / {:3d} | Reward: {:.3f} / {:.3f} | Loss: {:.3f} | Epsilon: {:.3f}'.format(agent.episode, step, total_step, reward, total_reward, loss, epsilon), end="")
            loss_values.append(loss)

            # max step for each episode is 1000
            # Keep track of every crucial info each episode
            if done or step > 320:
                write_to_csv(
                    episode=agent.episode, 
                    total_steps=total_step, 
                    loss_arr=loss_values, 
                    total_reward=total_reward, 
                    episode_score=info[0]['total_score'], 
                    csv_filepath=csv_path,
                    epsilon=epsilon
                )

            # Evaluate model for every given episode
                if agent.episode % 20 == 0:
                    print("\nSave Model ...")

                    agent.save_load_model(
                        op="save",
                        path=save_path,
                        fname=model_filename
                    )

                    
                    gif_name = f"train_ep" + str(agent.episode).zfill(5) + ".gif"
                    print(f"Generate GIF <{gif_name}>...")
                    img_buffer = play(env, agent, stack_frames, img_size, 0.50)
                    save_gif(img_buffer, gif_name)
                    print("Done !!")

                    if agent.episode % 400 == 0:
                        print("Doing backup...")
                        backup_filename = f"{model_filename}.ep{agent.episode}.qnet.bak"
                        orig_path = os.path.join(save_path, model_filename)
                        backup_path = os.path.join(save_path, backup_filename)
                        shutil.copy(orig_path, backup_path)
                        print(f"Backup done, file path: {backup_path}")

                agent.episode += 1
                break


            if total_step > max_steps:
                break

