'''
 Copyright 2021 Takahiro Kubo
https://github.com/icoxfog417/baby-steps-of-rl-ja

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http:www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

change log
Modificated GridworldEnv class to FootballEnv class
'''


import numpy as np
from gym.envs.toy_text import discrete
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


class FootballEnv(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, state, move_prob=0.8, default_reward=0.0):
        self.grid = [[0]*4 for i in range(11)] # 4列x11行
        self.grid[10][3] = 1 # t=4で10y以上進んでいたら報酬は1
        self.grid = np.array(self.grid)

        self._actions = [0,1,2,3,4]
        self.df = pd.read_csv("mdn_p.csv", index_col=0, header=0)

        self.default_reward = default_reward
        self.move_prob = move_prob

        num_states = self.nrow * self.ncol
        num_actions = len(self._actions)
        
        self.transition_probs_dict = {}

        # 初期状態はt=1の0y
        initial_state_prob = np.zeros(num_states)
        initial_state_prob[self.coordinate_to_state(0, 0)] = 1.0

        # Make transitions
        P = {}
        for s in range(num_states):
            if s not in P:
                P[s] = {}
            reward = self.reward_func(s)
            done = self.has_done(s)
            if done:
                # Terminal state
                for a in range(num_actions):
                    P[s][a] = []
                    P[s][a].append([1.0, s, reward, done])
            else:
                for a in range(num_actions):
                    P[s][a] = []
                    transition_probs = self.transit_func(s, a)
                    for n_s in transition_probs:
                        reward = self.reward_func(n_s)
                        done = self.has_done(n_s)
                        P[s][a].append([transition_probs[n_s], n_s,
                                        reward, done])
        self.P = P
        super().__init__(num_states, num_actions, P, initial_state_prob)

    @property
    def nrow(self):
        return self.grid.shape[0]

    @property
    def ncol(self):
        return self.grid.shape[1]

    @property
    def shape(self):
        return self.grid.shape

    @property
    def actions(self):
        return list(range(self.action_space.n))

    @property
    def states(self):
        return list(range(self.observation_space.n))

    def state_to_coordinate(self, s):
        # row, col = divmod(s, self.nrow)
        row, col = divmod(s, self.ncol)
        return row, col

    def coordinate_to_state(self, row, col):
        # index = row * self.nrow + col
        index = row * self.ncol + col
        return index

    def state_to_feature(self, s):
        feature = np.zeros(self.observation_space.n)
        feature[s] = 1.0
        return feature

    def transit_func(self, state, action):
        if (state,action) in self.transition_probs_dict.keys():
            return self.transition_probs_dict[(state,action)]
            
        transition_probs = {}

        # 現在の行と列
        row, col = self.state_to_coordinate(state)
        candidates = list(range(row, self.nrow)) # 次状態の行の候補 [n,n+1,...,10]

        # 選ばれた行動の確率遷移
        df = self.df[self.df["a"]==action] # 現在の行動はaction
        df = df[df["yard"]==row]           # 現在の距離はrow
        df = df[df["dn"]==col+1]           # 現在の距離はcol+1
        for i in candidates:
            next_state = self.coordinate_to_state(i, min(self.ncol-1, col+1))
            df_ = df[df["n_yard"]==i]
            transition_probs[next_state] = df_["p"].values[0]
            
        self.transition_probs_dict[(state,action)] = transition_probs
        return transition_probs

    def reward_func(self, state):
        row, col = self.state_to_coordinate(state)
        if (row >= self.nrow) or (col >= self.ncol): 
            return 0
        reward = self.grid[row][col]
        return reward

    def has_done(self, state):
        row, col = self.state_to_coordinate(state)
        reward = self.reward_func(state)
        if np.abs(reward) == 1:
            return True
        # 追加: 最終列でTrue
        elif col >= self.ncol-1:
            return True
        else:
            return False

    def _move(self, state, action):
        next_state = state
        row, col = self.state_to_coordinate(state)
        next_row, next_col = row, col

        # Move state by action
        if action == self._actions["LEFT"]:
            next_col -= 1
        elif action == self._actions["DOWN"]:
            next_row += 1
        elif action == self._actions["RIGHT"]:
            next_col += 1
        elif action == self._actions["UP"]:
            next_row -= 1

        # Check the out of grid
        if not (0 <= next_row < self.nrow):
            next_row, next_col = row, col
        if not (0 <= next_col < self.ncol):
            next_row, next_col = row, col

        next_state = self.coordinate_to_state(next_row, next_col)

        return next_state

    def plot_on_grid(self, values):
        if len(values.shape) < 2:
            values = values.reshape(self.shape)
        fig, ax = plt.subplots()
        ax.imshow(values, cmap=cm.RdYlGn)
        ax.set_xticks(np.arange(self.ncol))
        ax.set_yticks(np.arange(self.nrow))
        fig.tight_layout()
        #plt.show()
        plt.savefig('./reward.png')