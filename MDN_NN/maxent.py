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

import numpy as np
from planner import PolicyIterationPlanner
from tqdm import tqdm

expert_policy = [
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
]
expert_policy = np.array(expert_policy).reshape(44)



class MaxEntIRL():

    def __init__(self, env):
        self.env = env
        self.planner = PolicyIterationPlanner(env)
        self.actions = []
        self.thetas = []
        self.features = [np.zeros(44, dtype=float)]
        self.features[0][0] = 1

    def estimate(self, trajectories, epoch=20, learning_rate=0.01, gamma=0.9):
        # 特徴量行列 F
        # 16×16の単位行列
        state_features = np.vstack([self.env.state_to_feature(s)
                                   for s in self.env.states])
        # 1. パラメータθを初期化
        theta = np.random.uniform(size=state_features.shape[1])
        self.thetas.append(theta)
        # 2. エキスパートデータから特徴ベクトルに変換
        teacher_features = self.calculate_expected_feature(trajectories)
        self.teacher_features = teacher_features

        # 行動一致数
        for e in tqdm(range(epoch)):
            # Estimate reward.
            # 3. 状態ごとの報酬関数 R(s) = θ・F
            rewards = state_features.dot(theta.T)

            # 現時点のパラメータによる報酬関数を設定
            self.planner.reward_func = lambda s: rewards[s]
            # 4. 現時点の報酬関数に対して、方策を計算
            self.planner.plan(gamma=gamma)
            # print(self.planner.policy)

            # 5. 計算した方策で特徴ベクトルを取得
            features = self.expected_features_under_policy(
                                self.planner.policy, trajectories)
            # 6. 勾配を計算
            # μ_expert - μ(θ)
            update = teacher_features - features.dot(state_features)
            theta += learning_rate * update

            self.actions.append([l.tolist().index(1) for l in self.planner.policy])
            self.features.append(features.dot(state_features))
            self.thetas.append(theta)



        estimated = state_features.dot(theta.T)
        estimated = estimated.reshape(self.env.shape)
        return estimated

    def calculate_expected_feature(self, trajectories):
        '''
        エキスパートデータから特徴ベクトルを作成する関数

        :param trajectories: エキスパートの軌跡データ
        :return: 特徴ベクトル
        '''
        features = np.zeros(self.env.observation_space.n)
        for t in trajectories:
            for s in t:
                features[s] += 1

        features /= len(trajectories)
        return features

    def expected_features_under_policy(self, policy, trajectories):
        '''
        パラメータによる報酬関数から獲得される方策による特徴ベクトルの取得

        :param policy: 方策
        :param trajectories: エキスパート軌跡  軌跡数×状態のリスト
        :return: 各状態の頻度 16次元のリスト
        '''
        states = self.env.states
        P = self.env.P
        features = np.zeros_like(states, dtype=float)
        features[0] = 1 # 初期状態
        pre_features = self.features[-1]
        for s in states:
            # 現状態における最適行動を取得
            a = policy[s].tolist().index(1)
            # 次状態配列とその確率
            nexts = P[s][a]

            if s%4==3:
                continue
            for n in nexts:
                n_s = n[1]
                n_prob = n[0]
                features[n_s] += pre_features[s] * n_prob
        return features

irl = None
trajectories = None
def test_estimate():
    global irl
    global trajectories
    # 環境の設定
    from environment import FootballEnv
    env = FootballEnv()

    # エキスパートデータの収集
    trajectories = []
    env.seed(42)
    for i in range(10000):
        s = env.reset()
        done = False
        steps = [s]
        while not done:
            a = expert_policy[s]
            n_s, r, done, _ = env.step(a)
            steps.append(n_s)
            s = n_s
        trajectories.append(steps)
    # 訪問頻度
    p = np.zeros((11,4))
    for steps in trajectories:
        for s in steps:
            row, col = divmod(s, 4)
            p[row][col] += 1
    p /= len(trajectories)
    print(p)

    # 逆強化学習を実行
    print("Estimate reward.")
    irl = MaxEntIRL(env)
    rewards = irl.estimate(trajectories, epoch=500, gamma=1)
    env.plot_on_grid(rewards)
    print(rewards)

test_estimate()