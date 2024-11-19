# Notes about maze

### Loading Process

1. Load `maze1.txt` to `self.mazedata` (defined in `sprites.py`, `class MazeSprites(Spritesheet)`)
    * use `np.loadtxt` to load txt file directly into `Mazedata.data`

2. Initialize `self.mazedata` (defined in `mazedata.py`, `class MazeBase`)
    * `MazeData` works like metadata for each maze. It defines many important infos e.g. portal, pacman's start position etc. (refer to `mazedata.py`)

