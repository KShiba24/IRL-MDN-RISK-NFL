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
'''

from pickle import NONE
import numpy as np
import random


class Planner():

    def __init__(self, env, reward_func=None):
        self.env = env
        self.reward_func = reward_func
        if self.reward_func is None:
            self.reward_func = self.env.reward_func

    def initialize(self):
        self.env.reset()

    def transitions_at(self, state, action):
        reward = self.reward_func(state)
        done = self.env.has_done(state)
        transition = []
        if not done:
            transition_probs = self.env.transit_func(state, action)
            for next_state in transition_probs:
                prob = transition_probs[next_state]
                reward = self.reward_func(next_state)
                done = self.env.has_done(state)
                transition.append((prob, next_state, reward, done))
        else:
            transition.append((1.0, None, reward, done))
        for p, n_s, r, d in transition:
            yield p, n_s, r, d

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")


class ValuteIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        V = np.zeros(len(self.env.states))
        while True:
            delta = 0
            for s in self.env.states:
                expected_rewards = []
                for a in self.env.actions:
                    reward = 0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += p * (r + gamma * V[n_s] * (not done))
                    expected_rewards.append(reward)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        return V


class PolicyIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)
        self.policy = None
        self._limit_count = 1000

    def initialize(self):
        super().initialize()
        self.policy = np.ones((self.env.observation_space.n,
                               self.env.action_space.n))
        # First, take each action uniformly.
        self.policy = self.policy / self.env.action_space.n

    def policy_to_q(self, V, gamma):
        Q = np.zeros((self.env.observation_space.n,
                      self.env.action_space.n))

        for s in self.env.states:
            for a in self.env.actions:
                a_p = self.policy[s][a]
                for p, n_s, r, done in self.transitions_at(s, a):
                    if done:
                        Q[s][a] += p * a_p * r
                    else:
                        Q[s][a] += p * a_p * (r + gamma * V[n_s])
        return Q

    def estimate_by_policy(self, gamma, threshold):
        V = np.zeros(self.env.observation_space.n)

        count = 0
        while True:
            delta = 0
            for s in self.env.states:
                expected_rewards = []
                for a in self.env.actions:
                    action_prob = self.policy[s][a]
                    reward = 0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += action_prob * p * \
                                  (r + gamma * V[n_s] * (not done))
                    expected_rewards.append(reward)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value

            if delta < threshold or count > self._limit_count:
                break
            count += 1

        return V

    def act(self, s):
        # print(self.policy[s])
        return np.argmax(self.policy[s])

    def plan(self, gamma=0.9, threshold=0.0001, keep_policy=False):
        if not keep_policy:
            self.initialize()

        count = 0
        while True:
            update_stable = True
            # Estimate expected reward under current policy.
            V = self.estimate_by_policy(gamma, threshold)

            for s in self.env.states:
                # Get action following to the policy (choose max prob's action).
                policy_action = self.act(s)

                # Compare with other actions.
                action_rewards = np.zeros(len(self.env.actions))
                for a in self.env.actions:
                    reward = 0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += p * (r + gamma * V[n_s] * (not done))
                    action_rewards[a] = reward
                best_action = np.argmax(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                # Update policy (set best_action prob=1, otherwise=0 (greedy)).
                self.policy[s] = np.zeros(len(self.env.actions))
                self.policy[s][best_action] = 1.0

            if update_stable or count > self._limit_count:
                # If policy isn't updated, stop iteration.
                break
            count += 1

        return V

class QLearningPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)
        self.policy = None
        self._limit_count = 1000
        self.reward_func = None
        self.initialize()

    def initialize(self):
        super().initialize()
        self.policy = np.zeros((self.env.observation_space.n,
                               self.env.action_space.n))
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def act(self, s):
        Q_s = self.Q[s,:]
        return np.argmax(Q_s)

    def plan(self, gamma=0.9, threshold=0.0001, EPSILON = 1, keep_policy=False):
        EPOCH_MAX = 1000
        STEP_MAX  = 1000
        
        for epoch in range(EPOCH_MAX):
            s = self.env.reset() # 環境初期化
            step = 0 # ステップ数
            done = False # ゲーム終了フラグ
            total_reward = 0 # 累積報酬
            ALPHA = 0.5 * (EPOCH_MAX - epoch) / EPOCH_MAX

            while not done and step < STEP_MAX:
                # 行動選択
                if np.random.rand() > EPSILON:
                    # 最適な行動を予測
                    a = self.act(s)
                else:
                    a = random.choice(self.env.actions)

                # 行動
                n_s, reward, done, _ = self.env.step(a)
                n_a = self.act(n_s)
                update = (self.reward_func(s) + gamma * self.Q[n_s,n_a] * (not done))
                self.Q[s,a] = self.Q[s,a] + ALPHA * (update - self.Q[s,a])
                
        # 学習終了
        for s in range(self.env.observation_space.n):
            self.policy[s, self.act(s)] = 1