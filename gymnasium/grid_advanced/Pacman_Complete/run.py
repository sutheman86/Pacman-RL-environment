import pygame
import os
import numpy as np

from collections import deque

from pygame.locals import *
from .constants import *
from .pacman import Pacman
from .nodes import NodeGroup
from .pellets import Pellet, PelletGroup
from .ghosts import GhostGroup
from .fruit import Fruit
from .pauser import Pause
from .text import TextGroup
from .sprites import LifeSprites
from .sprites import MazeSprites
from .mazedata import MazeData
from .vector import Vector2

class Options:
    allowUserInput = False
    def __init__(self, allowUserInput=False):
        self.allowUserInput = allowUserInput


class GameController(object):
    def __init__(self, speedup=1.0, headless=False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

        self.debug_mode = False

        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.getready = True
        self.readytimer = 0
        self.readyTime = 3
        self.level = 0
        self.lives = 3
        self.score = 0

        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()
        self.assetpath = "assets/"
        self.clockCycle = (int)(1000/speedup)
        
        ### OBSERVATION: Grid-based States
        self.grid_state = np.zeros((NROWS, NCOLS))
        self.grid_state_position = np.zeros((NROWS, NCOLS))
        self.ret_state = np.zeros((NROWS, NCOLS))
        self.debug_print_timer = 0
        self.old_pacman_position = (0, 0)
        self.old_ghost_position = {}

        ### REWARD: Related attirbutes to calculating rewards

        # See MAZE.md.
        self.accessibletypes = ['.', '+', ' ', 'p', 'P', 'n', '-', '|']
        self.pellettypes = ['.', '+', 'p', 'P']

        self.reward = 0
        self.pacmanatepellet = 0
        self.pacmanwaiteatpellettimer = 0.0
        self.gobacktododgeghost = True
        ### REWARD_END
        np.set_printoptions(threshold=6969699, linewidth=6969699)

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):      
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.assetpath + self.mazedata.obj.name+".txt", self.assetpath + self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.assetpath + self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart))

        self.old_pacman_position = self.pacman.positionToGridCoord()

        self.pellets = PelletGroup(self.assetpath + self.mazedata.obj.name+".txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        
        for g in self.ghosts.ghosts:
            self.old_ghost_position[g.name] =  g.positionToGridCoord()

        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)
        self.readytimer = 0
        self.getready = True

        ### REWARD: Related attributes to calculate rewards
        self.lastmindisttoghost = np.inf
        self.escaping = 0
        self.mazebuffer = self.mazesprites.data.copy()
        ### REWARD_END

        ### OBSERVATION: Grid-state observation
        buf = np.array(self.mazebuffer)

        self.grid_state = np.zeros_like(buf, dtype=np.uint8)
        self.grid_state_position = np.zeros_like(buf, dtype=np.uint8)

        for i in range(buf.shape[0]):
            for j in range(buf.shape[1]):
                if buf[i][j].isdigit():
                    self.grid_state[i][j] = 254
                elif buf[i][j] in self.pellettypes:
                    self.grid_state[i][j] = 127

    ### OBSERVATION: Get Grid-State
    def getGameGridState(self):
        return self.ret_state.astype(np.uint8), self.grid_state_position

    def update(self):
        dt = self.clock.tick(30) / self.clockCycle
        self.textgroup.update(dt)
        self.pellets.update(dt)
        self.reward = 0
        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents(dt)
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.reward += self.pacman.update(dt)

                old_px, old_py = self.old_pacman_position
                self.grid_state_position[old_py][old_px] = 0

                px, py = self.pacman.positionToGridCoord()
                self.grid_state_position[py][px] = 192
                self.old_pacman_position = (px, py)

        else:
            self.pacman.update(dt)

        if self.getready and self.pause.paused:
            self.readytimer += dt
            if (int)(self.readytimer) == (int)(self.readyTime):
                self.pause = Pause(False)
                self.readytimer = 0
                self.getready = False
                self.textgroup.hideText()

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()

        distance = self.getMinDistanceFromGhosts()
        self.closeToGhost = (distance < 5)
        
        # REWARD_ADVANCED_3
        if not self.closeToGhost and self.reward <= 0:
            self.penalizeWalkingBackAndForth()
        # REWARD_END

        # REWARD_ADVANCED_2
        if self.closeToGhost:
            self.reward -= 8
        # REWARD_END

        # REWARD_ADVANCED_5
        if distance < 10 and distance > self.lastmindisttoghost:
            self.escaping += 1
            if self.escaping >= 30:
                self.reward += 2
        else:
            self.escaping = 0

        self.lastmindisttoghost = distance
        # REWARD_END

        self.checkEvents()
        self.render()

        self.debug_print_timer += dt

        # OBSERVATION
        self.ret_state = self.grid_state.copy()
        self.ret_state[self.grid_state_position != 0] = self.grid_state_position[self.grid_state_position != 0]

       #  if self.debug_print_timer >= 1.0 and self.debug_mode:
       #      os.system('clear')
       #      print(self.ret_state)
       #      self.debug_print_timer = 0.0


    # REWARD_ADVANCED_3
    def penalizeWalkingBackAndForth(self):
        direction = self.pacman.direction
        if direction == STOP:
            self.reward -= 5
            return
        opposite = -direction
        if opposite in self.pacman.facinghist:
            self.reward -= 5 # REWARD_END

    # REWARD_ADVANCED_2
    def getMinDistanceFromGhosts(self):
        px, py = self.pacman.positionToGridCoord()
        min = np.inf
        for ghost in self.ghosts.ghosts:
            cx, cy = ghost.positionToGridCoord()
            manhattan = np.abs(cx - px) + np.abs(cy - py)
            dist = manhattan
            if manhattan < 5:
                if 5 < min <= np.inf:
                    actual = self.getGhostPacmanDistance(ghost) 
                    if actual > manhattan:
                        dist = actual
            if dist < min:
                min = dist
        return min
    # REWARD_END

    # REWARD_BASIC_5 & REWARD_ADVANCED_2
    def getGhostPacmanDistance(self, ghost):
        px, py = self.pacman.positionToGridCoord()

        gx, gy = ghost.positionToGridCoord()

        queue = deque()
        queue.append(((px, py), 0))
        visited = set()
        visited.add(((px, py)))

        min_distance = 999999

        while queue:
            cur, dist = queue.popleft()
            
            if dist > 20:
                min_distance = 20
                break

            if (cur[0] == gx) and (cur[1] == gy):
                min_distance = dist
                break

            x = cur[0]
            y = cur[1]
            direction = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            for cx, cy in direction:
                if self.isInRange(cx, cy):
                    if self.isAccessible(cx, cy) and (cx, cy) not in visited:
                        queue.append(((cx, cy), dist+1))
                        visited.add((cx, cy))
        return min_distance
    # REWARD_END

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            #self.hideEntities()
    
    def checkPelletEvents(self, dt):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:

            self.pellets.numEaten += 1
            self.updateScore(pellet.points)

            # REWARD_BASIC_1:
            self.reward += 50
            # REWARD_END

            #REWARD_ADVANCED_1: calculate reward
            if not self.closeToGhost:
                self.reward += self.pacmanatepellet
                self.pacmanatepellet += 0.5
            # REWARD_END

            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)

            self.pellets.pelletList.remove(pellet)
            self.mazebuffer[pellet.my][pellet.mx] = ' '
            self.grid_state[pellet.my][pellet.mx] = 0.0


            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
                count = 0
                # REWARD_BASIC_5
                for g in self.ghosts.ghosts:
                    dist = self.getGhostPacmanDistance(g)
                    if dist < 5 and count < 2:
                        self.reward += 20
                # REWARD_END
                
            if self.pellets.isEmpty():
                self.flashBG = True

                # REWARD_BASIC_3: Finish Level (no pellets left)
                self.reward += 1000
                # REWARD_END

                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)
        else:
            # REWARD_ADVANCED_1:
            if self.pacmanwaiteatpellettimer >= 1.0:
                self.pacmanatepellet = 0
            else:
                self.pacmanwaiteatpellettimer += dt
            # REWARD_END

            # REWARD_ADVANCED_2: Close to pellets (if pacman isn't eating pellet)
            if not self.closeToGhost:
                dist = self.checkClosestPellet()
                if dist < 4:
                    self.reward += 2
            # REWARD_END
    
    # REWARD_ADVANCED_4
    def checkClosestPellet(self):
        px = int(self.pacman.position.x / TILEWIDTH)
        py = int(self.pacman.position.y / TILEHEIGHT)
        queue = deque()
        queue.append(((px, py), 0))
        visited = set()
        visited.add(((px, py)))

        min_distance = 999999

        while queue:
            cur, dist = queue.popleft()
            
            if dist > 20:
                min_distance = 20
                break

            if self.isPellet(cur[0], cur[1]):
                min_distance = dist
                break

            x = cur[0]
            y = cur[1]
            direction = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            for cx, cy in direction:
                if self.isInRange(cx, cy):
                    if self.isAccessible(cx, cy) and (cx, cy) not in visited:
                        queue.append(((cx, cy), dist+1))
                        visited.add((cx, cy))
        return min_distance
    # REWARD_END

    # REWARD_ADVANCED_4
    def isInRange(self, x, y):
        if (0 <= x < len(self.mazebuffer[0])) and (0 <= y < len(self.mazebuffer)): 
            return True
        return False
    # REWRAD_END

    # REWARD_ADVANCED_4
    def isAccessible(self, x, y):
        return self.mazebuffer[y][x] in self.accessibletypes
    # REWRAD_END

    # REWARD_ADVANCED_4
    def isPellet(self, x, y):
        return self.mazebuffer[y][x] in self.pellettypes
    # REWRAD_END
            

    def checkGhostEvents(self):
        for ghost in self.ghosts:

            # OBSERVATION: UPDATE POSITION OF EACH GHOST IN GRID_POSITION CHANNEL
            old_gx, old_gy = self.old_ghost_position[ghost.name]
            self.grid_state_position[old_gy][old_gx] = 0

            gx, gy = ghost.positionToGridCoord()
            self.grid_state_position[gy][gx] = 64
            self.old_ghost_position[ghost.name] = (gx, gy)

            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.reward += 100
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)                  
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1

                        # REWARD_BASIC_4: Pacman killed by ghost
                        self.reward -= 500
                        # REWARD_END

                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(pauseTime=3, func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)
    
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.reward += 0
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.lives = 3
        self.level = 0
        self.pause.paused = True
        self.getready = True
        self.readytimer = 0
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.getready = True
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        pygame.display.update()


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()



