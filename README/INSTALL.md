# ML Final Pacman environment

### Setup Virtual Environment

* First, make sure you have `miniconda` installed and set up.
    * [Miniconda Download Page](https://docs.anaconda.com/miniconda/miniconda-install/)

* clone this repository and enter the directory
    ```
    git clone git@github.com:sutheman86/Pacman-RL-environment.git && cd Pacman-RL-environment
    ```
    * ***Note:*** use `ssh` since this repo is not public. Make sure to set up your ssh key!! [Tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
    > or download this repo as zip file and extract it

* Create conda environment ***DON'T FORGET TO SPECIFY PYTHON VERSION TO 3.10***
    ```
    conda create -n pacman-RL-environment python=3.10
    ```

* activate this conda environment
    ```
    conda activate pacman-RL-environment
    ```

* after created and activated the environment, current environment name should be:
    ```
    (pacman-RL-environment)
    ```

### Install Packages and gymnasium environment

* Use conda to install `pytorch`
    ```
    conda install pytorch
    ```

* Install required packages using pip
    ```
    pip install gym-notices gymnasium matplotlib pandas numpy
    ```
    * note that `pygame` package is already installed as requirement for `gymnasium`
