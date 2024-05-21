from board import Board 
from model import * 

def generateGreedyAction(network, state, possible_actions_mask):
    action_values = network(state)
    mask = torch.zeros(TOTAL_ACTIONS)
    base_value = 1000000
    mask[possible_actions_mask] = base_value
    action = torch.argmax(action_values + mask)
    return int(action), action_values[action]
