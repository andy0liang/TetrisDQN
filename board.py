import numpy as np
import random
from block import Block

class Board:
    def __init__(self, boardWidth=10, boardHeight=20):
        self.matrix = np.zeros((boardHeight, boardWidth))
        self.boardWidth = boardWidth
        self.boardHeight = boardHeight
        self.height = 0
        self.score = 0
        self.currentBlock = None
        self.heldBlock = None
        self.blockQueue = []
        self.canSwap = True
        self.cycleBlock()

    def resetBlockQueue(self):
        blocks = [Block(i) for i in range(Block.NUM_SHAPES)]
        random.shuffle(blocks)
        self.blockQueue = blocks

    def getNextBlock(self):
        return self.blockQueue[-1]

    def cycleBlock(self):
        if not self.blockQueue:
            self.resetBlockQueue()
        self.currentBlock = self.blockQueue.pop()
        if not self.blockQueue:
            self.resetBlockQueue()
        self.canSwap = True

    def swapBlock(self):
        if not self.canSwap:
            return
        if self.heldBlock is not None:
            self.heldBlock, self.currentBlock = self.currentBlock, self.heldBlock
            self.currentBlock.resetRotation()
            self.canSwap = False
        else:
            self.heldBlock = self.currentBlock
            self.cycleBlock()

    def rotateBlock(self):
        self.currentBlock.rotate()

    def dropBlock(self, column):
        lastSuccessRow = -1
        while self.isLegalMove(lastSuccessRow + 1, column):
            lastSuccessRow += 1
        if lastSuccessRow == -1:
            return False
        self._placeBlock(lastSuccessRow, column)
        self.cycleBlock()
        return True

    def _placeBlock(self, x, y):
        for dx, dy in self.currentBlock.getCoords():
            x2 = x + dx
            y2 = y + dy
            self.matrix[x2][y2] = 1

    def isLegalMove(self, x, y):
        for dx, dy in self.currentBlock.getCoords():
            x2 = x + dx
            y2 = y + dy
            if not self.isInBounds(x2, y2):
                return False
            if self.matrix[x2][y2] == 1:
                return False
        return True
    
    def isInBounds(self, x, y):
        return not (x < 0 or y < 0 or x >= self.boardHeight or y >= self.boardWidth)

    def __str__(self):
        r, c = self.matrix.shape
        res = ""
        for i in range(r):
            for j in range(c):
                if(self.matrix[i][j] == 1):
                    res += '■ '
                else:
                    res += '. '
            res += '\n'
        res += "=================================\n"
        res += f'Current block: {self.currentBlock}\n'
        res += f'Next block: {self.getNextBlock()}\n'
        res += f'Held block: {self.heldBlock}\n'
        res += "=================================\n"
        return res
