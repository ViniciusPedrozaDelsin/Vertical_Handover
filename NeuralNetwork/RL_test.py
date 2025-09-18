import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
from scipy.stats import beta

# --------------------------
# 1. Environment definition
# --------------------------
class HandoverEnv:
    def __init__(self, n_poa=15, episode_length=100):
        self.n_poa = n_poa
        self.episode_length = episode_length
        self.state_dim = n_poa
        self.means = [2, 1, 2, 3, 1, 1, 2, 1, 2, 4, 1, 2, 3, 5, 1]
        self.reset()
    
    def calculateBeta(self, means, alfa_param=5, beta_param=5):
        dist = beta(alfa_param, beta_param)
        samples = []
        i = 0
        for mean in means:
            sample = dist.rvs(len(means))[i] * (mean/0.5)
            samples.append(sample)
            i += 1
            #print(f"Number {i} | Expected Mean {mean} | Sample {sample}")
        return samples
        
    def reset(self):
        self.t = 0
        # Random initial signals
        #self.state = np.random.rand(self.state_dim)
        self.state = np.array(self.calculateBeta(self.means), dtype=np.float32)
        self.rmse_history = []
        return self.state

    def step(self, action):
        self.t += 1
        # Fake RMSE contribution = difference between best PoA and chosen one
        best_signal = np.max(self.state)
        chosen_signal = self.state[action]
        rmse = abs(best_signal - chosen_signal)
        self.rmse_history.append(rmse)

        # Update state randomly (device moves)
        #self.state = np.random.rand(self.state_dim)
        self.state = np.array(self.calculateBeta(self.means), dtype=np.float32)

        reward = -rmse
        #reward = 0.0
        
        done = self.t >= self.episode_length
        '''if done:
            # Final reward = negative RMSE
            final_rmse = np.sqrt(np.mean(np.square(self.rmse_history)))
            reward = -final_rmse'''
        return self.state, reward, done, {}

# --------------------------
# 2. DQN Agent
# --------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.05, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0
        self.epsilon_decay = 0.97
        self.batch_size = 64
        self.memory = deque(maxlen=8192)

        # Build Q-network
        self.model = self.build_model(lr)

    def build_model(self, lr):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def replay(self, ep):
        
        if ep < 64:
            times = 1
        elif ep >= 64 and ep < 112:
            times = 2
        else:
            times = 5
        
        #print("============== Start Replay ==============")
        for _ in range(times):
            if len(self.memory) < self.batch_size:
                return
            minibatch = random.sample(self.memory, self.batch_size)
            
            #minibatch = self.memory
            # Get Min Reward
            #r =  min(transition[2] for transition in self.memory)
            
            states, targets = [], []
            for s, a, r, s_next, done in minibatch:
                #print("===== '' =====")
                #print(f"State: {s}")
                #print(f"Action: {a}")
                #print(f"Reward: {r}")
                #print(f"Next State: {s_next}")
                #print("===== '' =====")
                target = r + (self.gamma * r)
                if not done:
                    #q_next = np.max(self.model.predict(s_next[np.newaxis], verbose=0)[0])
                    q_next = self.model.predict(s_next[np.newaxis], verbose=0)[0][a]
                    target = r + (self.gamma * q_next)
                #print(f"Target: {target}")
                #print("===== '' =====")

                q_vals = self.model.predict(s[np.newaxis], verbose=0)[0]
                q_vals[a] = target

                states.append(s)
                targets.append(q_vals)

            self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --------------------------
# 3. Training Loop
# --------------------------
env = HandoverEnv(n_poa=15, episode_length=128)
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.n_poa)

n_episodes = 128
for ep in range(n_episodes):
    state = env.reset()
    total_reward = 0
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        #print(f"State {state}")
        #print(f"Action {action}") 
        #print(f"Next_State {next_state}")
        #print(f"Reward {reward}")
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            agent.replay(ep)
            print(f"Episode {ep+1}/{n_episodes}, Reward: {total_reward:.3f}, Epsilon: {agent.epsilon:.2f}")
            break
