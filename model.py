import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from board import Board 

TOTAL_ACTIONS = 13 # swap/rotate/9 drop

LRStart = 1e-2
GAMMA = .99 
BOARD_GRID_SIZE = 10 * 20 
CURRENT_BLOCK_SIZE = 7
HELD_BLOCK_SIZE = 8
NEXT_BLOCK_SIZE = 7 
BOOL_CAN_SWAP = 1
CURRENT_BLOCK_ROT = 4



class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(BOARD_GRID_SIZE + CURRENT_BLOCK_SIZE + HELD_BLOCK_SIZE+NEXT_BLOCK_SIZE + BOOL_CAN_SWAP +CURRENT_BLOCK_ROT, 128), nn.ReLU(), 
                  nn.Linear(128,64), nn.ReLU(), 
                  nn.Linear(64,32), nn.ReLU(), 
                  nn.Linear(32,16), nn.ReLU(), 
                  nn.Linear(16,TOTAL_ACTIONS)]
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = LRStart)
        
    def forward(self, input_state):
        return self.model(input_state)
    
    def backprop(self, state, action, true_value):
        self.optimizer.zero_grad()
        computed_value = torch.sum(self.model(state) * action)
        mse_loss = (computed_value - true_value) ** 2
        mse_loss.backward()
        self.optimizer.step()

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
