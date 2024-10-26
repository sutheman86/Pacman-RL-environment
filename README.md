# Pacman-RL-environment

### Installation:  [See README/install.md](./README/INSTALL.md)

### Known issues:

* Weird in-game bug:
    * Sometimes it appears that ghost and pacman collides, but nothing happened.
    * Pacman will skip eating pellets when:
        1. `speedup` is too big (it should be less than `6.0` if frame rate = `30`
        2. External pause (I will explain it)

* Matplotlib interrupts pygame logic
    * `plt.pause(time)` caused pacman to skip pellets.
        * There's no solution to this problem yet, temporary solution is setting `time < 1 / framerate`.
        > Maybe try to separate `env` logic with in-game logic???

### TODOs

- [ ] load **trained** `pytorch` model (`foo.pt`) into gymnasium environment
- [ ] integrate the algorithm, allow training.
- [ ] provide jupiter notebook for algorithm research group to use.
    - [ ] build environment for gymnasium to load properly
    - [ ] combine DQN algorithm from `pacman-test.ipynb` to `pacman-world.ipynb`

