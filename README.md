# Pacman-RL-environment

### Reward Design: [See src/MYSTRAT.md](./src/MYSTRAT.md)

### Installation:  
* **[See README/install.md](./README/INSTALL.md)**

### Ongoing Experiments

#### Method
* I'll train each model for 8~9 hours
* Then I'll evaluate the model for 10 times, taking the mean score (not reward). I'll record the success rate (pacman passed first level)

- [ ] (Working) Compare **Dueling Double DQN** and **Double DQN**
    > Reward keeps the same.

* For all of the experiments below, I'll pick Dueling or no-Dueling DQN depends on the performance.

* Major changes, experiments must be done
- [ ] Compare With **Close To Pellets** Metric and without it.
- [ ] Compare With **Scare Ghosts away** and without it.
- [ ] Compare With **Binary Logic** and without it.
- [ ] Compare with **Reward Move back** and without it.
    
* Minor changes, I'll apply it directly if there's no time to experiment it.
- [ ] Compare with using **Manhatton distance** and **Hybrid metric**


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
- [ ] ~~make `Pacman_Complete` to return necessary info for `pacman_world.py` to calculate reward.~~
    > It turns out it's better to calculate reward in the game, maybe I'll change my mind later...
    
- [ ] ~~Implement multi-agent environment (we want to train ghost and pacman at the same time.)~~ ***Finish Training Pacman First***
    > This can be done by modifying `gymnasium` environment, but we'll switch too `pettingzoo` if needed. (Which provides similar interface to `gymnasium`)

#### Training

- [ ] implement `Curiosity`
- [x] Use `Double DQN`, which is better than the original one.
- [x] Use `Dueling DQN`
    > maybe combine them into `Double Dueling DQN`?
- [ ] Experiment with Rewards
    > (from sutheman86: I have my custom reward implemented in `sutheman` branch)

#### Apply Curriculum Learning method
    
* Because it's too hard for agent to play well with original settings (4 ghosts, the performance is close to random with 200000 steps)

* So now we want to split the process into the easier ones and training agent to finish each of them.

- [x] Stage 1. Without ghost, pacman should finish the game in 800~1000 steps
    > Done. he could finish the game in ~1200 steps

- [ ] Stage 2. Add one ghost into the game:
    - [ ] Stage 2.1: the ghost is half of its original speed.
        > Almost finished it within 160000 training steps.
    - [ ] Stage 2.2: ghost is in its original speed.

- [ ] etc. etc. Finish these two jobs first

* We might add ghost agents (unlikely) if:
    1. Stage 1. is satisfied.
    2. Multi-agent environment is implemented.
