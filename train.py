from model import *
from util import *
import random

epsilon = 1
iterations = 10000
minibatch_size = 10
copy_model_freq = 10
decrement_lr_freq = 100
model_save_freq = 1000
decrement_eps_freq = 10

DQNetwork = DQN()
TargetNetwork = DQN()
TargetNetwork.load_state_dict(DQNetwork.state_dict())
TargetNetwork.eval()


score_history = np.array([])
best_move_history = np.array([])
move_history = np.array([])

experienceReplayBuffer = []

for iteration in range(1, iterations + 1):
    board = Board()

    while not board.isTerminal():
        state = board.getState()
        legalMoves = board.getLegalMoves()

        # Epsilon Greedy
        if random.random() <= epsilon:
            action = random.choice(legalMoves)
        else:
            with torch.no_grad():
                action, predicted_value = generateGreedyAction(DQNetwork, state, board.getPossibleActionsMask())
        
        reward = board.makeAction(action)
        newState = board.getState()

        experienceReplayBuffer.append((state, action, reward, newState))

        if len(experienceReplayBuffer) < minibatch_size:
            continue

        samples = random.sample(experienceReplayBuffer, minibatch_size)

        states, actions, rewards, newStates = zip(*samples)

        states = torch.tensor(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)

        current_q_values = DQNetwork(states).gather(1, actions).squeeze(1)

        next_q_values = TargetNetwork(next_states).max(1)[0]
        expected_q_values = rewards + (GAMMA * next_q_values)

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        DQNetwork.optimize(loss)

    if iteration % copy_model_freq == 0:
        TargetNetwork.load_state_dict(DQNetwork.state_dict())


