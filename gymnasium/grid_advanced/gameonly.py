import Pacman_Complete.run as run

game = run.GameController()
run.Options(allowUserInput=True)
game.startGame()

while True:
    game.update()
