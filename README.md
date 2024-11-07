# Pacman-RL-environment

### Installation:  
* **[See README/install.md](./README/INSTALL.md)**

### Changes done on `Pacman_Complete`

Here are ***GAME LOGIC*** changes done for training.

* Ghost Group: In `ghosts.py` `class GhostGroup(object)`
    * Original: `self.ghosts = [self.blinky, self.inky, self.clyde, self.pinky]`
    * Now: `self.ghosts = [self.blinky]`

* Ghost speed: In `ghosts.py`, right now it's **half of its original speed**.
    * `__init__`: add `self.setSpeed(50)`, originally its 100.
    * `reset(self)`: changed to `self.setSpeed(50)`, originally it's `self.setSpeed(100)`
    * `startFreight(self)`: changed to `self.setSpeed(25)`, originally it's `self.setSpeed(25)`
    * `normalMode(self)`: changed to `self.setSpeed(50)`, originally it's `self.setSpeed(100)`



### TODOs

#### Environment

- [X] load **trained** `pytorch` model (`foo.pt`) into gymnasium environment
- [x] integrate the algorithm, allow training.
- [x] provide jupiter notebook for algorithm research group to use.
    - [x] build environment for gymnasium to load properly
    - [x] combine DQN algorithm from `pacman-test.ipynb` to `pacman-world.ipynb`
- [x] provide script to train and evaluate model conveniently
    > to train model, run `python train.py` (or `!python train.py` in colab notebook.)
- [ ] make `Pacman_Complete` to return necessary info for `pacman_world.py` to calculate reward.
- [ ] ~~Implement multi-agent environment (we want to train ghost and pacman at the same time.)~~ ***Finish Training Pacman First***
    > This can be done by modifying `gymnasium` environment, but we'll switch too `pettingzoo` if needed. (Which provides similar interface to `gymnasium`)

#### Training

- [ ] implement `Curiosity`
- [ ] Use `Double DQN`, which is better than the original one.

#### Apply Curriculum Learning method
    
* Because it's too hard for agent to play well with original settings (4 ghosts, the performance is close to random with 200000 steps)

* So now we want to split the process into the easier ones and training agent to finish each of them.

- [ ] Stage 1. Without ghost, pacman should finish the game in 800~1000 steps
    > We don't know what's the minimum required time to walk through the whole maze. So the number might be too small.

- [ ] Stage 2. Add one ghost into the game
    > We realized that even with one ghost, it's still too difficult for pacman. So we will nerf ghost's speed first.

- [ ] etc. etc. Finish these two jobs first

* We might add ghost agents if:
    1. Stage 1. is satisfied.
    2. Multi-agent environment is implemented.
