from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import Pacman_Complete as pmac

from Pacman_Complete.constants import *
from Pacman_Complete.pacman import Pacman
from Pacman_Complete.nodes import NodeGroup
from Pacman_Complete.pellets import PelletGroup
from Pacman_Complete.ghosts import GhostGroup
from Pacman_Complete.fruit import Fruit
from Pacman_Complete.pauser import Pause
from Pacman_Complete.text import TextGroup
from Pacman_Complete.sprites import LifeSprites
from Pacman_Complete.sprites import MazeSprites
from Pacman_Complete.mazedata import MazeData

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    stop = 4

class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.level = 0
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()

    
    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm
