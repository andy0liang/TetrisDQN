from model import *
from util import *
from collections import deque
import random
import matplotlib.pyplot as plt
import os

epsilon = 1
iterations = 10000
minibatch_size = 512
copy_model_freq = 100 # couple hundred or couple thousands
decrement_lr_freq = 100
model_save_freq = 1000
decrement_eps_freq = 10
print_board_freq = 500
print_reward_freq = 100

DQNetwork = DQN()
TargetNetwork = DQN()

# checkpoint_path = ''
# DQNetwork.load_state_dict(torch.load(checkpoint_path))

TargetNetwork.load_state_dict(DQNetwork.state_dict())
TargetNetwork.eval()


score_history = np.array([])
best_move_history = np.array([])
move_history = np.array([])

if not os.path.exists('models'):
    os.makedirs('models')

experienceReplayBuffer = deque(maxlen=20000)

rewards_per_iteration = []

for iteration in range(1, iterations + 1):
    board = Board()
    totalReward = 0
    while not board.isTerminal():
        state = board.getState()
        legalMoves = board.getLegalMovesInts()

        # Epsilon Greedy
        if random.random() <= epsilon:
            action = random.choice(legalMoves)
        else:
            with torch.no_grad():
                action, predicted_value = generateGreedyAction(DQNetwork, state, board.getPossibleActionsMask())
        
        reward = board.makeAction(action)
        newState = board.getState()

        totalReward += reward

        experienceReplayBuffer.append((state, action, reward, newState))

        if len(experienceReplayBuffer) < minibatch_size:
            continue

        samples = random.sample(experienceReplayBuffer, minibatch_size)

        states, actions, rewards, next_states = zip(*samples)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))

        current_q_values = DQNetwork(states).gather(1, actions).squeeze(1)

        next_q_values = TargetNetwork(next_states).max(1)[0]
        expected_q_values = rewards + (GAMMA * next_q_values)

        loss = nn.HuberLoss()(current_q_values, expected_q_values)
        # Huber loss

        DQNetwork.optimize(loss)

    if iteration % 2 == 0 and epsilon > 0.01:
        epsilon *= 0.995
        epsilon = max(epsilon, 0.01)

    rewards_per_iteration.append(totalReward)
    print(f"Iteration {iteration}/{iterations}, Total Reward: {totalReward}, Tetris Score {board.score}")
    if iteration % print_reward_freq == 0 or iteration == 1:
        print(f"Iteration {iteration}/{iterations}, Total Reward: {totalReward}, Tetris Score {board.score}")
        if len(rewards_per_iteration) < 50:
            mean = sum(rewards_per_iteration) / len(rewards_per_iteration)
        else:
            mean = sum(rewards_per_iteration[-50:]) / 50
        print(f"Running mean of last 50: {mean}")

    if iteration % model_save_freq == 0:
        torch.save(DQNetwork.state_dict(), f'models/dqn_checkpoint_{iteration}.pth')

    if iteration % copy_model_freq == 0:
        TargetNetwork.load_state_dict(DQNetwork.state_dict())
    
    if iteration % print_board_freq == 0:
        print(f"Iteration {iteration}:")
        print(board)

torch.save(DQNetwork.state_dict(), 'models/dqn_final.pth')


plt.plot(rewards_per_iteration)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.show()

