import Pacman_Complete.run as run

game = run.GameController()

game.startGame()

while True:
    game.update()

