import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F
import os

# Set random seeds for reproducibility
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Hyperparameters
        self.gamma = 0.99            # Discount factor
        self.epsilon = 1.0           # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.target_update_freq = 1000  # Update target network every 1000 steps

        # Replay memory
        self.memory = ReplayMemory(capacity=10000)

        # Q-Network
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is in evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

        # Step counter for updating target network
        self.steps_done = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Shape: (1, state_size)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = self.memory.sample(self.batch_size)

        # Prepare batches
        states = torch.FloatTensor([sample[0] for sample in minibatch]).to(self.device)         # Shape: (batch_size, state_size)
        actions = torch.LongTensor([sample[1] for sample in minibatch]).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)
        rewards = torch.FloatTensor([sample[2] for sample in minibatch]).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)
        next_states = torch.FloatTensor([sample[3] for sample in minibatch]).to(self.device)    # Shape: (batch_size, state_size)
        dones = torch.FloatTensor([sample[4] for sample in minibatch]).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)  # Shape: (batch_size, 1)

        # Next Q values from target network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)  # Shape: (batch_size, 1)

        # Compute target Q values
        target_q = rewards + (1 - dones) * self.gamma * max_next_q  # Shape: (batch_size, 1)

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()

def train_dqn(agent, env, episodes, device):
    max_treward = 0
    for e in range(1, episodes + 1):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False
        trunc = False
        while not done and not trunc:
            action = agent.act(state)
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = np.array(next_state)
            reward = reward if not done else -10  # Penalize if episode ends
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += 1  # Increment by 1 per step

        if total_reward > max_treward:
            max_treward = total_reward

        if e % 10 == 0:
            print(f"Episode: {e}/{episodes}, Score: {total_reward}, Max Score: {max_treward}, Epsilon: {agent.epsilon:.2f}")

    print("Training completed.")

def test_dqn(agent, env, episodes, device):
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Disable exploration
    for e in range(1, episodes + 1):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False
        trunc = False
        while not done and not trunc:
            action = agent.act(state)
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = np.array(next_state)
            state = next_state
            total_reward += 1
        print(f"Test Episode: {e}, Score: {total_reward}")
    agent.epsilon = original_epsilon  # Restore original epsilon

def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment
    env = gym.make('CartPole-v1', render_mode=None)

    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, device)

    # Pre-populate replay memory with random experiences
    print("Pre-populating replay memory...")
    for _ in range(agent.batch_size):
        state, _ = env.reset()
        state = np.array(state)
        action = random.randrange(action_size)
        next_state, reward, done, trunc, _ = env.step(action)
        next_state = np.array(next_state)
        reward = reward if not done else -10
        agent.remember(state, action, reward, next_state, done)
        if done or trunc:
            continue

    # Train the agent
    print("Starting training...")
    train_dqn(agent, env, episodes=500, device=device)

    # Save the trained model
    agent.save("dqn_cartpole.pth")

    # Test the agent
    print("Starting testing...")
    test_dqn(agent, env, episodes=10, device=device)

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()

