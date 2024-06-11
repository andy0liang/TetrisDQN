import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from board import *
import os


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 20 * 10, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x[:, :200].view(-1, 1, 20, 10) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
    

class WCDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, alpha_start=1.0, alpha_end=0.1, alpha_decay=0.995, checkpoint_path="models/wcdqn_checkpoint.pth"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay

        self.policy_net = PolicyNetwork(state_dim, 3)  # Outputs alpha, beta, gamma
        self.q_net_1 = QNetwork(200, action_dim)  # Note: Input dim should match the input to the conv layers
        self.q_net_2 = QNetwork(200, action_dim)
        self.q_net_3 = QNetwork(200, action_dim)
        self.target_q_net_1 = QNetwork(200, action_dim)
        self.target_q_net_2 = QNetwork(200, action_dim)
        self.target_q_net_3 = QNetwork(200, action_dim)
        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict())
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict())
        self.target_q_net_3.load_state_dict(self.q_net_3.state_dict())
        
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_q1 = optim.Adam(self.q_net_1.parameters(), lr=lr)
        self.optimizer_q2 = optim.Adam(self.q_net_2.parameters(), lr=lr)
        self.optimizer_q3 = optim.Adam(self.q_net_3.parameters(), lr=lr)

        self.memory = []
        self.batch_size = 64

        self.N1 = 1
        self.N2 = 1
        self.N3 = 1
        self.t1 = 1
        self.t2 = 1
        self.t3 = 1

        self.checkpoint_path = checkpoint_path
        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.q_net_1.load_state_dict(checkpoint['q_net_1_state_dict'])
            self.q_net_2.load_state_dict(checkpoint['q_net_2_state_dict'])
            self.q_net_3.load_state_dict(checkpoint['q_net_3_state_dict'])
            self.target_q_net_1.load_state_dict(checkpoint['target_q_net_1_state_dict'])
            self.target_q_net_2.load_state_dict(checkpoint['target_q_net_2_state_dict'])
            self.target_q_net_3.load_state_dict(checkpoint['target_q_net_3_state_dict'])
            self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
            self.optimizer_q1.load_state_dict(checkpoint['optimizer_q1_state_dict'])
            self.optimizer_q2.load_state_dict(checkpoint['optimizer_q2_state_dict'])
            self.optimizer_q3.load_state_dict(checkpoint['optimizer_q3_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.alpha = checkpoint['alpha']
            self.memory = checkpoint['memory']
            print("Checkpoint loaded successfully")

    def save_checkpoint(self):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'q_net_1_state_dict': self.q_net_1.state_dict(),
            'q_net_2_state_dict': self.q_net_2.state_dict(),
            'q_net_3_state_dict': self.q_net_3.state_dict(),
            'target_q_net_1_state_dict': self.target_q_net_1.state_dict(),
            'target_q_net_2_state_dict': self.target_q_net_2.state_dict(),
            'target_q_net_3_state_dict': self.target_q_net_3.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'optimizer_q1_state_dict': self.optimizer_q1.state_dict(),
            'optimizer_q2_state_dict': self.optimizer_q2.state_dict(),
            'optimizer_q3_state_dict': self.optimizer_q3.state_dict(),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'memory': self.memory
        }
        torch.save(checkpoint, self.checkpoint_path)
        print("Checkpoint saved successfully")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert state to tensor and add batch dimension
            alpha_beta_gamma = self.policy_net(state_tensor)
            alpha, beta, gamma = alpha_beta_gamma[0]
            
            q_values_1 = self.q_net_1(state_tensor)
            q_values_2 = self.q_net_2(state_tensor)
            q_values_3 = self.q_net_3(state_tensor)
            
            c1 = (alpha * self.t1) / (self.N1 * ((alpha / self.N1) + (beta / self.N2) + (gamma / self.N3)))
            c2 = (beta * self.t2) / (self.N2 * ((alpha / self.N1) + (beta / self.N2) + (gamma / self.N3)))
            c3 = (gamma * self.t3) / (self.N3 * ((alpha / self.N1) + (beta / self.N2) + (gamma / self.N3)))
            
            c_max = max(c1, c2, c3)
            
            if c_max == c1:
                return q_values_1.argmax().item()
            elif c_max == c2:
                return q_values_2.argmax().item()
            else:
                return q_values_3.argmax().item()

    def store_transition(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)), 
                torch.tensor(actions), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.FloatTensor(np.array(next_states)), 
                torch.tensor(dones, dtype=torch.float32))

    def expert_reward(self, state, action):
        return 0

    def combined_reward(self, state, action, reward):
        expert_reward = self.expert_reward(state, action)
        combined_reward = self.alpha * expert_reward + (1 - self.alpha) * reward
        return combined_reward

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        q_values_1 = self.q_net_1(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_values_2 = self.q_net_2(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_values_3 = self.q_net_3(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        next_q_values_1 = self.target_q_net_1(next_states).max(1)[0]
        next_q_values_2 = self.target_q_net_2(next_states).max(1)[0]
        next_q_values_3 = self.target_q_net_3(next_states).max(1)[0]

        combined_rewards = torch.tensor([self.combined_reward(states[i], actions[i], rewards[i]) for i in range(len(rewards))], dtype=torch.float32)

        target_q_values_1 = combined_rewards + (1 - dones) * self.gamma * next_q_values_1
        target_q_values_2 = combined_rewards + (1 - dones) * self.gamma * next_q_values_2
        target_q_values_3 = combined_rewards + (1 - dones) * self.gamma * next_q_values_3

        loss_1 = F.mse_loss(q_values_1, target_q_values_1)
        loss_2 = F.mse_loss(q_values_2, target_q_values_2)
        loss_3 = F.mse_loss(q_values_3, target_q_values_3)

        self.optimizer_q1.zero_grad()
        loss_1.backward()
        self.optimizer_q1.step()

        self.optimizer_q2.zero_grad()
        loss_2.backward()
        self.optimizer_q2.step()

        self.optimizer_q3.zero_grad()
        loss_3.backward()
        self.optimizer_q3.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.alpha = max(self.alpha_end, self.alpha * self.alpha_decay)

    def update_target(self):
        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict())
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict())
        self.target_q_net_3.load_state_dict(self.q_net_3.state_dict())


num_episodes = 1000
update_target_every = 10

board = Board()
agent = WCDQNAgent(state_dim=227, action_dim=15)

for episode in range(num_episodes):
    state = board.getState()
    done = False

    while not done:
        action = agent.select_action(state)
        reward = board.makeAction(action)
        
        next_state = board.getState()
        done = done or board.isTerminal()

        agent.store_transition((state, action, reward, next_state, done))
        state = next_state

        if done:
            break

    agent.update_policy()
    if episode % update_target_every == 0:
        agent.update_target()
        agent.save_checkpoint()

    print(f"Episode {episode} finished with score: {board.score}")

print("Training completed.")


