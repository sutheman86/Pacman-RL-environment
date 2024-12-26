# Pacman-RL-environment

### Installation:  

1. Setup Virtual Environment

* First, make sure you have conda installed and set up.

* clone this repository and enter the directory
    ```
    git clone https:/github.com/sutheman86/Pacman-RL-environment && cd Pacman-RL-environment
    ```

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

2. Install Packages and gymnasium environment

* Use conda to install `pytorch`
    ```
    conda install pytorch
    ```

* Install required packages using pip
    ```
    pip install gym-notices gymnasium matplotlib pandas==2.1.4 numpy
    ```
    * note that `pygame` package is already installed as requirement for `gymnasium`
    * lock `panda` to lower version because latest version `2.2.3`has weird bugs.

3. Run setup script (interactive) for your experimental needs.
    ```
    python setup.py
    ```

4. depends on your preferences, use script or notebook to start training!
    * Script: `src`, `train.py` for training; `eval.py` for evaluation.
    * Jupyter Notebook: inside `notebooks`, check the variant you want.

5. To change settings, run `setup.py` again to set up easily.
