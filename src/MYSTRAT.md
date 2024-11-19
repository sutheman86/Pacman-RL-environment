# My Reward Strategy

Most rewards are calculated in `Pacman_Complete/run.py`, to see the code related to such reward, search `REWARD` when tracing code.
> e.g. search `REWARD_BASIC_1` to find code about calculate eaten pellet reward.

### Basic Rewards: `REWARD_BASIC`

* `REWARD_BASIC_1` &rarr; **eaten pellet**: `100` for each pellet

* <i><s>**eaten power pellets**: `120` for each pellet</i></s> 
    * **Should be replaced with scare ghost away reward**
        > Pacman should only eat pellet when ghosts are nearby.

* `REWARD_BASIC_2` &rarr; **eaten ghost** : `100` for each ghost

* `REWARD_BASIC_3` &rarr; **Killed**: `-500` for each death

* <s>**Game over**: `-1000` for each occurence</s>
    > No need to calculate this anymore.

* `REWARD_BASIC_4` &rarr; **Not getting any reward**: `-2` for each tick

##### `REWARD_BASIC_5`: Scare Ghosts away

> reference: [paper](https://cs229.stanford.edu/proj2017/final-reports/5241109.pdf)

* Ghosts will try to run away from pacman when enetering `FREIGHT` mode.
* So Pacman can also "escape" from ghosts by eating power pellet when ghosts are nearby.

If pacman ate any power pellets:
1. Check distance between Pacman and all ghosts.
2. for each ghost:
    * if ghost is within a certain radius, agent receives a reward.
    * The reward is given **2** times maximum. (Scare 2 ghosts away at most)

* `REWARD_BASIC_6`: Incorrect input (agent's direction of choice collides with walls)
    * penalize `0.5` points for each time.
    * Please go to `pacman.py` to see it.

### Extra Reward: `REWARD_ADVANCED`

#### `REWARD_ADVANCED_1`: **Pellet Bonus**

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

#### `REWARD_ADVANCED_2`: Distance

* If pacman is too close to any of the ghosts, a penalty is received

* **Grid**: record position information
    * The whole maze is represented as grid, it's stored in `self.mazesprites.data`. I copy it to `self.mazebuffer` for edit.
    * Grid position of any object can be transformed from pixel position.
        ```python
        pacman.col = int(pacman.position.x / TILEWIDTH)
        pacman.row = int(pacman.position.x / TILEHEIGHT)
        ```
    * There's some error when rounded to `int`, but it's close enough (8 pixels at most)


* **Distance Metric**: Manhattan for estimation, use BFS if close.
    $$ Distance_{Manhattan} \le Distance_{actual}$$
    * &rarr; Manhattan would only **underestimate** distance between two nodes

    * Manhattan formula:
        $$|x_a - x_b| + |y_a - y_b|$$
    * BFS: please refer to code. If `dist > 5`, return 10 directly (anything &gt; 5), 
        * doing so could prevent spending too much time calculate uneccessary value.

* **Threshold**: `5` tiles.

* the penalty value is `3` for each tick
    > So if pacman is too close with Blinky for 1 second, he will be penalized for `90` points`

#### `REWARD_ADVANCED_3`: Wondering without Reward

* Pacman will receive penalty when he moves back &amp; forth and receive no reward.
* A bonus penalty of `2` points will be received for each tick.
* a `deque` is used to keep track of previous movements
    * This `deque` has size of `30`, which track for every movement in 1 seconds
    * it's in `pacman.py`, `self.facinghist` in `class Pacman`

* **Exception**: Moving to escape ghost (If ghost were too close, it makes sense to give up rewards).

#### `REWARD_ADVANCED_4`: Pellet Distance

> I found out that `cKDTree` is not required for this process

* If pellet were close enough to pacman within certain threshold, then agent receive `2` points each tick.

* **Threshold**: `4` grid distance.

* Method: again, use **BFS**, setting maximum distance to prevent uneccessary calculation.

#### `REWARD_ADVANCED_5`: Escape From Ghosts

* By comparing distance of last tick and this tick. This makes things easier.
* `self.escaping`: A counter counting continuous time period of pacman escaping from the ghosts.
* If pacman stay escaing for one second or more (`self.escaping` > 30), reward `2` points for each tick.
* This only works if pacman is decently close to ghosts (`10` grid distance)
