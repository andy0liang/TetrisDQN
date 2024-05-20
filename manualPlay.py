from board import Board

currentBoard = Board()
while True:
    print(currentBoard)
    action = input("Action (d/s/r/q): ").strip()
    if action == 'q':
        break
    elif action == 'r':
        currentBoard.rotateBlock()
    elif action == 's':
        currentBoard.swapBlock()
    elif action == 'd':
        column = int(input('Column: ').strip())
        result = currentBoard.dropBlock(column)
        if not result:
            print(" ----- ILLEGAL MOVE ----- ")
        