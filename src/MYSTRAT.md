# My Reward Strategy

### Normal Reward

* Eaten pellet: `100` for each pellet
* Eaten Power pellet: `120` for each pellet
* Eaten ghost: `100` for each ghost

### Normal Penalty

* Killed: `-500` for each death
* Game over: `-1000` for each occurence
* Not getting any reward: `-2` for each tick


### Extra Reward/Penalty

#### **Pellet Bonus**: 

* For each pellet eaten, an counter will increase by 1.
* When eaten pellet, reward will be added by this counter value
* Example: (`<` refers to pacman `*` refers to pellet to be eaten)
    ```
    <  *   *   *   *   *   *   *
      100 100 100 100 100 100 100 (Base Reward)
        0   1   2   3   4   5   6 (Bonus)
      100 101 102 103 104 105 106 (Actal Reward Received)
    ```
* The counter is `self.pacmanatepellet`
* The counter will reset to 0 when no pellet is eaten for `1` second. (if no speed up is applied)

#### Distance

* If pacman is too close to any of the ghosts, a penalty is received

* Distance: Use *Manhattan Distance* as metric
    > Cannot use Euclidean Distance as metric because Pacman and Ghost cannot move diagnolly
    * Formula:
    $$D = |x_{pacman} - x_{ghost}| + |y_{pacman} - y_{ghost}|$$

* Threshold: `60` pixel
* I really want to use $A^{*}$ as metric, but right now I don't have time to implement it. And I'm afraid it would be too computationally expensive.

* the penalty value is `3` for each tick
    > So if pacman is too close with Blinky for 1 second, he will be penalized for `90` points`

#### Wondering without Reward

* Pacman will receive penalty when he moves back &amp; forth and receive no reward.
* A bonus penalty of `2` points will be received for each tick.
* a `deque` is used to keep track of previous movements
    * This `deque` has size of `30`, which track for every movement in 1 seconds
    * it's in `pacman.py`, `self.facinghist` in `class Pacman`

* **Exception**: Moving to escape ghost (If ghost were too close, it makes sense to give up rewards).

### TO-DOs

* If the reward change is major, we might have to train model from scratch.


#### Minor Changes


##### Better Distance Metric

* Use A* to calculate distance.
* Will use Manhattan Distance to roughly estimate distance, then if rough distance is too close. Use $A^{*}$ to calculate exact distance

#### Major Changes

##### Escape From Ghosts

* If pacman is too close to any ghost, moving opposite direction gets reward.
    * If moving to opposite direction for one second (30 frame) could get rid of any ghost, then reward is given

##### Pellet Distance

> The cKDTree part is implemented, but reward hasn't been calculated yet.

* If pellet were close enough to pacman within certain threshold, then agent receive `2` points each tick.

* Implementation: Use `cKDTree` from `scipy` (You'll need to install `scipy`)

Steps

1. keep all pellets in a list (`self.availablepellets`)
2. build cKDTree `self.pellettree` based on `self.availablepellets`
3. for each tick:
    1. if a pellet is eaten, then rebuild the cKDTree
        > It won't lead to any noticable performance drop, don't worry.
    2. if not:
        * get closest pellet to pacman
        * calculate its manhattan distance
        * if the distance is within a certain radius, agent receive reward.

##### Scare Ghosts away

> reference: [paper](https://cs229.stanford.edu/proj2017/final-reports/5241109.pdf)

* Ghosts will try to run away from pacman when enetering `FREIGHT` mode.
* So Pacman can also "escape" from ghosts by eating power pellet when ghosts are nearby.

If pacman ate any power pellets:
1. Check distance between Pacman and all ghosts.
2. for each ghost:
    * if ghost is within a certain radius, agent receives a reward.
    * The reward is given **2** times maximum.


##### Apply Binary Logic

* Combine *Ghost is nearby* and *Pellet is nearby*
