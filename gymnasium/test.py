import gymnasium_env
import gymnasium
import warnings

# 忽略 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 初始化環境
env = gymnasium.make('gymnasium_env/PacmanGymEnv', speedup=5.0)
obs, info = env.reset()

# 設定最多 100 步
for step in range(1000000):  
    action = env.action_space.sample()  # 隨機取樣一個動作
    obs, reward, done, info = env.step(action)

    
    # 顯示當前觀察、獎勵及步數
    env.render()
    #print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Done: {done}")

    # 檢查是否回合結束
    if done:
        print("Episode finished!")
        break

# 關閉環境
env.close()
