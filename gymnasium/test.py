import gymnasium_env
import gymnasium
import warnings
from PIL import Image
import numpy as np
import os

# 設定無視窗模式
os.environ["SDL_VIDEODRIVER"] = "dummy"

# 忽略 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 初始化環境
env = gymnasium.make('gymnasium_env/PacmanGymEnv', speedup=4.0)
obs, info = env.reset()
env_unwrapped = env.unwrapped   

# 用來保存每一幀的圖像
frames = []

# 設定最多 1000 步
for step in range(1000):
    action = env.action_space.sample()  # 隨機取樣一個動作
    obs, reward, done, info = env.step(action)
    
    
    # 確保返回值非空並且是 numpy array 格式
    if obs is not None and isinstance(obs, np.ndarray):
        img = Image.fromarray(obs)  # 轉換為 PIL 圖像格式
        frames.append(img)  # 添加幀到 frames 列表中
    else:
        print("Render did not return a valid image.")
    
    print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Info: {info}, Done: {done}")

    # 檢查是否回合結束
    if done:
        print("Episode finished!")
        break

# 關閉環境
env.close()

# 保存為 GIF
output_path = "../Gif/pacman_game.gif"
frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=1)
print(f"GIF saved as {output_path}")
