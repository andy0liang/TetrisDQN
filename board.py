import numpy as np
import random
import torch
import torch.nn.functional as F
from block import Block

ILLEGAL_MOVE_REWARD = -50
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
        # blocks = [Block(1)]
        self.blockQueue = blocks

    def clearRowsIfPossible(self):
        count = 0
        while np.all(self.matrix[-1] == 1):
            self._clearRow()
            count += 1
        self.rowsCleared += count
        self.score += CLEAR_ROW_SCORES[count]

    def getLegalMoves(self):
        # (swap, rotate, drop, column)
        legals = [(0, 1, 0, 0)]
        if self.canSwap:
            legals.append((1, 0, 0, 0))
        for column in range(-3, 10):
            if self.isLegalMove(0, column):
                legals.append((0, 0, 1, column))
        return legals

    def getLegalMovesInts(self):
        mask = self.getPossibleActionsMask()
        return torch.nonzero(mask).squeeze(1).tolist()

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
            return False
        if self.heldBlock is not None:
            self.heldBlock, self.currentBlock = self.currentBlock, self.heldBlock
            self.currentBlock.resetRotation()
            self.canSwap = False
        else:
            self.heldBlock = self.currentBlock
            self.cycleBlock()
        return True

    # def reward(self, action):
    #     # (swap, rotate, drop, column)
    #     swap, rotate, drop, column = action

    #     if swap != 0:
    #         return 0
    #     if rotate != 0:
    #         return 0
        
    #     totalReward = 0

    #     level = (self.rowsCleared // 10) + 1

    #     if not self.isLegalMove(0, column):
    #         return ILLEGAL_MOVE_REWARD * level
        
    #     matrix2 = self.matrix.copy()
    #     lastSuccessRow = -1
    #     while self.isLegalMove(lastSuccessRow + 1, column):
    #         lastSuccessRow += 1
    #     if lastSuccessRow == -1:
    #         return ILLEGAL_MOVE_REWARD * level
    #     for dx, dy in self.currentBlock.getCoords():
    #         x2 = lastSuccessRow + dx
    #         y2 = column + dy
    #         matrix2[x2][y2] = 1

    #     count = 0
    #     while np.all(matrix2[-1] == 1):
    #         matrix2 = np.vstack([np.zeros(matrix2.shape[1]), matrix2[:-1]])
    #         count += 1
    #     totalReward += CLEAR_ROW_SCORES[count] * level

    #     return totalReward

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
    
    def getState(self):
        '''
        Tuple {
            matrix: np.ndarray
            currentBlock: one-hot
            currentBlockRotation: one-hot
            heldBlock: one-hot
            nextBlock: one-hot 
            canSwap: boolean (0 or 1)
        }, length = 20*10 + 7 + 4 + 8 + 7 + 1 = 223
        '''

        # Flatten the game matrix
        flattenedMatrix = self.matrix.flatten()

        # One-hot encode the current block
        currentBlockOneHot = np.zeros(Block.NUM_SHAPES)
        currentBlockOneHot[self.currentBlock.block_id] = 1

        # One-hot encode the current block's rotation
        currentBlockRotationOneHot = np.zeros(4)
        currentBlockRotationOneHot[self.currentBlock.rotation] = 1

        # One-hot encode the held block
        heldBlockOneHot = np.zeros(Block.NUM_SHAPES + 1)
        if self.heldBlock is not None:
            heldBlockOneHot[self.heldBlock.block_id] = 1
        else:
            heldBlockOneHot[-1] = 1

        # One-hot encode the next block in the queue
        nextBlockOneHot = np.zeros(Block.NUM_SHAPES)
        nextBlockOneHot[self.blockQueue[-1].block_id] = 1

        # Encode whether the player can swap blocks
        canSwapOneHot = np.zeros(1)
        if self.canSwap:
            canSwapOneHot[0] = 1

        # Concatenate all the arrays into one long numpy array
        state_array = np.concatenate([
            flattenedMatrix,
            currentBlockOneHot,
            currentBlockRotationOneHot,
            heldBlockOneHot,
            nextBlockOneHot,
            canSwapOneHot
        ])

        return state_array

    def makeAction(self, action):
        # Takes in just an int
        # (swap, rotate, -2, -1, 0, 1, 2, 3...)
        if self.isTerminal():
            return 0
        
        if action == 0:  # swap
            if self.swapBlock():
                return 0
            else:
                return ILLEGAL_MOVE_REWARD * ((self.rowsCleared // 10) + 1)
        
        if action == 1:  # rotate
            self.rotateBlock()
            return 0
        
        column = action - 4
        oldScore = self.score
        heuristic = self.calculate_heuristic() * 0.2
        if self.dropBlock(column):
            return self.score - oldScore + heuristic + 5
        else:
            return ILLEGAL_MOVE_REWARD * ((self.rowsCleared // 10) + 1)

    def getPossibleActionsMask(self):
        actions = torch.zeros(13, dtype=torch.long)

        if self.canSwap:
            actions[0] = 1
        
        # rotate
        actions[1] = 1

        for column in range(-2, 9):
            if self.isLegalMove(0, column):
                actions[column + 4] = 1

        return actions


    def calculate_heuristic(self):
        holes = self.calculate_holes()
        bumpiness = self.calculate_bumpiness()
        highest_height = self.calculate_highest_height()
        aggregate_height = self.calculate_aggregate_height()
        wells = self.calculate_wells()
        
        # Example heuristic calculation (tune the weights as needed)
        heuristic = -1 * highest_height - 0.2 * aggregate_height - 0.25 * holes - 0.6 * bumpiness #- 0.2 * wells
        
        return heuristic

    def calculate_holes(self):
        holes = 0
        for col in range(self.boardWidth):
            block_found = False
            for row in range(self.boardHeight):
                if self.matrix[row][col] != 0:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def calculate_bumpiness(self):
        heights = [self.calculate_column_height(col) for col in range(self.boardWidth)]
        bumpiness = sum(abs(heights[col] - heights[col + 1]) for col in range(len(heights) - 1))
        return bumpiness

    def calculate_highest_height(self):
        return max(self.calculate_column_height(col) for col in range(self.boardWidth))

    def calculate_aggregate_height(self):
        return sum(self.calculate_column_height(col) for col in range(self.boardWidth))

    def calculate_wells(self):
        wells = 0
        for col in range(self.boardWidth):
            for row in range(self.boardHeight):
                if self.matrix[row][col] == 0:
                    if (col == 0 or self.matrix[row][col - 1] != 0) and (col == self.boardWidth - 1 or self.matrix[row][col + 1] != 0):
                        wells += 1
        return wells

    def calculate_column_height(self, col):
        for row in range(self.boardHeight):
            if self.matrix[row][col] != 0:
                return self.boardHeight - row
        return 0