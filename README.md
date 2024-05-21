# TetrisDQN

Game State:

Tuple {
    matrix: np.ndarray
    currentBlock: one-hot
    currentBlockRotation: one-hot
    heldBlock: one-hot
    nextBlock: one-hot 
    canSwap: boolean (0 or 1)
}

Board: {
    board_matrix: np.ndarray
    score: int
    height: int
    ...
}

Block: {
    block_id: int
    rotation: int
}

