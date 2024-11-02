import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from Pacman_Complete.run import GameController
from Pacman_Complete.run import Options
from Pacman_Complete.constants import *

class PacmanGymEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    def __init__(self, speedup=1.0, render_mode='rgb_array'):
        super(PacmanGymEnv, self).__init__()
        
        # 設定動作空間: 上、下、左、右
        self.action_space = spaces.Discrete(5)  # 例如：0=上, 1=下, 2=左, 3=右
        
        # 設定觀察空間: 將遊戲畫面作為觀察
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(SCREENHEIGHT, SCREENWIDTH, 3), dtype=np.uint8)
        
        # 初始化遊戲控制器
        if render_mode == 'rgb_array':
            self.game = GameController(speedup=speedup,headless=True)
        else:
            self.game = GameController(speedup=speedup)
        self.render_mode = render_mode
        self.options = Options(allowUserInput=False)
        self.game.startGame()
        self.speedup = speedup
    
    def reset(self, seed=None, options=None):
        """重置環境並返回初始觀察值"""
        super().reset(seed=seed)  # 設定隨機種子（如有必要）
        self.game.restartGame()
        observation = self._get_observation()
        return observation, {}
    
    def step(self, action):
        """根據動作更新遊戲狀態並返回觀察值、回報、完成狀態、額外資訊"""
        
        # 根據動作選擇方向
        directions = [STOP, UP, DOWN, LEFT, RIGHT]
        if action in range(5):
            self.game.pacman.facing = directions[action]
        
        # 更新遊戲狀態
        self.game.update()
        
        # 獲取觀察值、回報和完成狀態
        observation = self._get_observation()
        done = (self.game.lives == 0)  # Pac-Man 被吃掉時遊戲結束
        reward = self.game.reward  # 使用當前得分作為回報
        
        info = {"lives":self.game.lives,"total_score":self.game.score}
        
        return observation, reward, done, info
    
    def render(self):
        """顯示遊戲畫面"""
        if self.render_mode == 'human':
            self.game.render()
        if self.render_mode == 'rgb_array':
            return self._get_observation()
    
    def _get_observation(self):
        """提取當前畫面並返回觀察數據"""
        # 從遊戲畫面中提取圖像資料
        observation = pygame.surfarray.array3d(self.game.screen)
        observation = np.transpose(observation, (1, 0, 2))  # 調整為 Gym 格式
        return observation
