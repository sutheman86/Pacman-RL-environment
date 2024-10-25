# ML Final Pacman environment

### Setup Virtual Environment

* First, set up python `venv`.
    ```
    python -m venv .
    ```

* Activate the virtual environment (in the project root)
    * On Windows:
    ```
    ./Scripts/Activate
    ```
    * On mac/Linux:
    ```
    source ./bin/activate
    ```

### Install Packages and gymnasium environment

* Install required packages
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
