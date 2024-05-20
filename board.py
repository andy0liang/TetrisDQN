import numpy as np
import random
from block import Block

ILLEGAL_MOVE_REWARD = -5000
CLEAR_ROW_SCORES = [0, 100, 300, 500, 800]

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
        self.rowsCleared = 0
        self.cycleBlock()

    def resetBlockQueue(self):
        blocks = [Block(i) for i in range(Block.NUM_SHAPES)]
        random.shuffle(blocks)
        self.blockQueue = blocks

    def clearRowsIfPossible(self):
        count = 0
        while np.all(self.matrix[-1] == 1):
            self._clearRow()
            count += 1
        self.rowsCleared += count
        self.score += CLEAR_ROW_SCORES[count]

    def _clearRow(self):
        self.matrix = np.vstack([np.zeros(self.matrix.shape[1]), self.matrix[:-1]])

    def isTerminal(self):
        for i in range(3, 7):
            if not self.isLegalMove(0, i):
                return True
        return False

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

    def reward(self, action):
        # (swap, rotate, drop, column)
        swap, rotate, drop, column = action

        if swap != 0:
            return 0
        if rotate != 0:
            return 0
        
        totalReward = 0

        level = (self.rowsCleared // 10) + 1

        if not self.isLegalMove(0, column):
            return ILLEGAL_MOVE_REWARD * level
        
        matrix2 = self.matrix.copy()
        lastSuccessRow = -1
        while self.isLegalMove(lastSuccessRow + 1, column):
            lastSuccessRow += 1
        if lastSuccessRow == -1:
            return ILLEGAL_MOVE_REWARD * level
        for dx, dy in self.currentBlock.getCoords():
            x2 = lastSuccessRow + dx
            y2 = column + dy
            matrix2[x2][y2] = 1

        count = 0
        while np.all(matrix2[-1] == 1):
            matrix2 = np.vstack([np.zeros(matrix2.shape[1]), matrix2[:-1]])
            count += 1
        totalReward += CLEAR_ROW_SCORES[count] * level

        return totalReward

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
        self.clearRowsIfPossible()
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
        for i in range(self.boardWidth):
            res += str(i) + (" " * (2-len(str(i))))
        res += '\n'
        res += "=================================\n"
        res += f'Current block: {self.currentBlock}\n'
        res += f'Next block: {self.getNextBlock()}\n'
        res += f'Held block: {self.heldBlock}\n'
        res += f'Can swap: {self.canSwap}\n'
        res += f'Current score: {self.score}\n'
        res += f'Current level: {self.rowsCleared // 10 + 1}\n'
        res += "=================================\n"
        return res
