import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from environment import MountainCar

class LinearModel:
    def __init__(self, state_size, action_size, lr, indices):
        self.lr = lr
        self.indices = indices
        self.weights = np.zeros((state_size + 1, action_size))

    def vectorize(self, state):
        if self.indices:
            s = [0.0 for i in range(2048)]
            for i in range(2048):
                if i in state: s[i] = 1
        else:
            s = list(state.values())
        return np.concatenate((np.array([1]), np.array(s)))
                
    def predict(self, state):
        s = self.vectorize(state)
        return (self.weights.T).dot(s)  
    
    def update(self, state, action, target):
        s = self.vectorize(state)
        q = (self.weights.T[action]).dot(s)
        self.weights.T[action] -= self.lr * (q - target) * s
        for i in range(3):
            if (i != action):
                self.weights[0][i] = self.weights[0][action]

class QlearningAgent:
    def __init__(self, env: MountainCar, mode: str = None, gamma: float = 0.9,
                 lr: float = 0.01, epsilon: float = 0.05):
        self.env = env
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.returns = []
        self.r_means = []
        if (self.mode == "raw"):
            self.lm = LinearModel(env.state_space, 3, lr, False)
        elif (self.mode == "tile"):
            self.lm = LinearModel(env.state_space, 3, lr, True)
    
    def get_action(self, state):
        qs = (self.lm).predict(state)
        optimal = np.argmax(np.array(qs))
        probs = [0.0 for i in range(3)]
        for i in range(3):
            if (i == optimal):
                probs[i] = (1 - self.epsilon) + self.epsilon / 3
            else:
                probs[i] = self.epsilon / 3
        action = random.choices(range(3), weights = probs)[0]
        return action

    def train(self, episodes, max_iterations):
        returns = []
        for i in range(episodes):
            iters = 0
            r = 0.0
            s = self.env.transform(self.env.state)
            while iters < max_iterations:
                action = self.get_action(s)
                new_s, reward, done = self.env.step(action)
                r += reward
                target = reward + self.gamma * np.max((self.lm).predict(new_s))
                self.lm.update(s, action, target)
                if done: break
                s = new_s
                iters += 1
            returns.append(r)
            self.env.reset()
        self.returns = returns

    def get_means(self):
        r_means = []
        for start in range(len(self.returns)):
            if (start + 25) <= len(self.returns):
                end = start + 25
            else:
                end = len(self.returns)
            r_mean = np.mean(np.array(self.returns[start:end]))
            r_means.append(r_mean)
        self.r_means = r_means

    def plot(self):
        x = list(range(1, 401))
        plt.plot(x, self.returns, label = "returns")
        plt.plot(x, self.r_means, label = "rolling means")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()

    def write_returns(self, r_out):
        with open(r_out, 'w+') as outfile:
            for num in self.returns:
                outfile.write(str(num) + '\n')
            outfile.close()
    
    def write_weights(self, w_out):
        with open(w_out, 'w+') as outfile:
            flattened = (self.lm.weights).flatten()
            for num in flattened[2:]:
                outfile.write(str(num) + '\n')
            outfile.close()

def main(args):
    mode = args[1]
    w_out = args[2]
    r_out = args[3]
    episodes = int(args[4])
    max_iters = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    lr = float(args[8])
    return mode, w_out, r_out, episodes, max_iters, epsilon, gamma, lr

if __name__ == "__main__":
    mode, w_out, r_out, episodes, max_iters, epsilon, gamma, lr = main(sys.argv)
    env = MountainCar(mode=mode)
    agent = QlearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)
    agent.train(episodes, max_iters)
    agent.write_returns(r_out)
    agent.write_weights(w_out)
    agent.get_means()
    agent.plot()

    #py q_learning.py tile fixed_weight.out fixed_returns.out 400 200 0.05 0.99 0.00005

