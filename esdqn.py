import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import DBSCAN
from board import Board
import random
import os

class ESDQN(nn.Module):
    def __init__(self, input_dim, action_dim, learning_rate=1e-3):
        super(ESDQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def update(self, states, actions, targets):
        self.optimizer.zero_grad()
        outputs = self(states)
        outputs = outputs.gather(1, actions.unsqueeze(1)).squeeze(1)  # Select the Q-values for the taken actions
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

class ElasticStepDQN:
    def __init__(self, input_dim, action_dim, learning_rate=1e-3, gamma=0.99, checkpoint_path="models/esdqn_checkpoint.pth"):
        self.q_network = ESDQN(input_dim, action_dim, learning_rate)
        self.target_network = ESDQN(input_dim, action_dim, learning_rate)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.experience_replay = []
        self.batch_size = 64
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        self.dbscan = DBSCAN(eps=3, min_samples=2)
        self.checkpoint_path = checkpoint_path
        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.q_network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.experience_replay = checkpoint['experience_replay']
            print("Checkpoint loaded successfully")

    def save_checkpoint(self):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.q_network.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'experience_replay': self.experience_replay
        }
        torch.save(checkpoint, self.checkpoint_path)
        print("Checkpoint saved successfully")

    def select_action(self, state, legal_moves):
        if random.random() <= self.epsilon:
            return random.choice(legal_moves)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
            legal_q_values = q_values[legal_moves]
            return legal_moves[torch.argmax(legal_q_values).item()]

    def store_experience(self, experience):
        self.experience_replay.append(experience)
        if len(self.experience_replay) > 10000:
            self.experience_replay.pop(0)

    def sample_experiences(self):
        indices = np.random.choice(len(self.experience_replay), self.batch_size, replace=False)
        return [self.experience_replay[i] for i in indices]

    def cluster_states(self, states):
        if len(states) < 2:
            return [0] * len(states)
        return self.dbscan.fit_predict(states)

    def compute_targets(self, experiences):
        states, actions, rewards, next_states = zip(*experiences)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * max_next_q_values)

        return states, actions, expected_q_values

    def train_step(self):
        if len(self.experience_replay) < self.batch_size:
            return

        experiences = self.sample_experiences()
        states, actions, expected_q_values = self.compute_targets(experiences)

        state_clusters = self.cluster_states([exp[0] for exp in experiences])
        for cluster_id in np.unique(state_clusters):
            cluster_indices = np.where(state_clusters == cluster_id)[0]
            self.q_network.update(
                states[cluster_indices],
                actions[cluster_indices],
                expected_q_values[cluster_indices]
            )

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, iterations=1000, update_target_frequency=10):
        for iteration in range(iterations):
            board = Board()
            while not board.isTerminal():
                state = board.getState()
                legal_moves = board.getLegalMovesInts()
                action = self.select_action(state, legal_moves)
                reward = board.makeAction(action)
                next_state = board.getState()

                self.store_experience((state, action, reward, next_state))

                self.train_step()

                if self.epsilon > self.min_epsilon:
                    self.epsilon *= self.epsilon_decay

            if iteration % update_target_frequency == 0:
                self.update_target_network()
                print(f"Target network updated at iteration {iteration}")
                self.save_checkpoint()

if __name__ == "__main__":
    input_dim = 227  # Board state size
    action_dim = 13  # Number of possible actions
    esdqn_agent = ElasticStepDQN(input_dim, action_dim)
    esdqn_agent.train(iterations=10000)
