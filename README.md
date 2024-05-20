# TetrisDQN

Game State:

Tuple {
    Board: Board
    Current block: Block
    Next block: Block 
    Held block: Block
}

Board: {
    board_matrix: np.ndarray
    score: int
    height: int
}

Block: {
    block_id: int
    rotation: int
}

