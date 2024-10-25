# ML Final Pacman environment

### Setup Virtual Environment

* First, make sure you have `miniconda` installed and set up.
    * [Miniconda Download Page](https://docs.anaconda.com/miniconda/miniconda-install/)

* clone this repository and enter the directory
    ```
    git clone git@github.com:sutheman86/Pacman-RL-environment.git && cd Pacman-RL-environment
    ```
    * ***Note:*** use `ssh` since this repo is not public. Make sure to set up your ssh key!! [Tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

* create conda environment `env` ***inside*** repo directory.
    ```
    conda create -p ./env
    ```

* activate this conda environment
    ```
    conda activate ./env
    ```

* after created and activated the environment, current environment name should be:
    ```
    (/path/to/Pacman-RL-environment/env)
    ```

### Install Packages and gymnasium environment

* **Install pip first** using conda
    ```
    conda install pip
    ```

* Install required packages using pip
    ```
    pip install babel copier Flask-Caching gym-notices gymnasium typing
    ```
    * note that `pygame` package is already installed as requirement for `gymnasium`

* go to `gymnasium` directory
    ```
    cd gymnasium
    ```

* install `gymnasium_env` environment
    ```
    pip install -e .
    ```

### Test

* To test model training, run:
    ```
    python test.py
    ```

* To *run pacman only* , run:
    ```
    python gameonly.py
    ```
