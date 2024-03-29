import gym
import time
import random
import warnings

warnings.filterwarnings("ignore")

env = gym.make("GridWorld-v0")   #创建环境名

class Learn(object):
    def __init__(self, grid_mdp):     # 初始化状态值函数
        self.v = dict()
        for state in grid_mdp.states:
            self.v[state] = 0



        # 初始化策略，这些策略均在状态转移概率矩阵中
        self.pi = dict()
        # random.choice(seq):返回列表、元组、字符串的随机项
        self.pi[1] = random.choice(['e', 's'])       #往这几个能走的方向随便走
        self.pi[2] = random.choice(['e', 'w'])
        self.pi[3] = random.choice(['w', 's', 'e'])
        self.pi[4] = random.choice(['e', 'w'])
        self.pi[5] = random.choice(['w', 's'])

    # 策略迭代函数（包括策略评估和策略改善）
    def policy_iterate(self, grid_mdp):
    # 迭代100次直到策略不变为止
        for i in range(100):
            # 策略评估和策略改善交替进行
            self.policy_evaluate(grid_mdp)
            self.policy_improve(grid_mdp)

    # 策略评估：
    def policy_evaluate(self, grid_mdp):
    # 迭代1000次计算每个状态的真实状态值函数
        for i in range(1000):
            delta = 0.0
            # 遍历状态空间
            for state in grid_mdp.states:
                # 终止状态不用计算状态值函数（v=0.0）
                if state in grid_mdp.terminate_states:
                    continue
                action = self.pi[state]
                t, s, r = grid_mdp.transform(state, action)    #r就是刚刚说过的回报
                new_v = r + grid_mdp.gamma * self.v[state]    #初始化的v是0，状态值函数（有点没明白，那不是加的上一步的状态值函数吗）
                delta += abs(new_v - self.v[state])       #状态值函数的变化
                # 更新状态值函数
                self.v[state] = new_v
            if delta < 1e-6:   #基本上没啥变化了
                break
                #我的理解就是，哪个回报更大，哪个策略被选中的概率就越高

    # 策略改善:遍历动作空间，寻找最优动作
    def policy_improve(self, grid_mdp):
        # 在每个状态下采用贪婪策略
        for state in grid_mdp.states:
            # 终止状态不用计算状态值函数(v=0.0)和求最优策略
            if state in grid_mdp.terminate_states:
                continue
            # 假设当前策略为最优动作
            a1 = self.pi[state]  # 上面不是随便选了嘛，就当他很好
            t, s, r = grid_mdp.transform(state, a1)
            v1 = r + grid_mdp.gamma * self.v[state]
            # 遍历动作空间与最优动作进行比较，从而找到最优动作
            for action in grid_mdp.actions:
                # 当不在状态转移概率中时，状态动作值函数不存在，状态值函数不变
                t, s, r = grid_mdp.transform(state, action)
                if s != -1:  # 就是不是原地打转的时候
                    if v1 < r + grid_mdp.gamma * self.v[s]:  # 因为r变了，历遍了嘛
                        a1 = action
                        v1 = r + grid_mdp.gamma * self.v[s]
            # 更新策略
            self.pi[state] = a1

    # 最优动作

    def action(self, state):
        return self.pi[state]

gm = env.env
# 初始化智能体的状态
state = env.reset()
# 实例化对象，获得初始化状态值函数和初始化策略
learn = Learn(gm)
# 策略评估和策略改善
learn.policy_iterate(gm)
total_reward = 0
# 最多走100步到达终止状态
for i in range(100):
    env.render()
    # 每个状态的策略都是最优策略
    action = learn.action(state)
    # 每一步按照最优策略走
    state, reward, done, _ = env.step(action)
    total_reward += reward
    time.sleep(1)
    if done:
        # 显示环境中物体进入终止状态的图像
        env.render()
        break

def main():
    env = gym.make("GridWorld-v0")
    gm = env.env
    state = env.reset()
    learn = Learn(gm)

            # count = 0
            # while True:
            #
            #     env.render()
            #     time.sleep(1)
            #
            #     action = env.action_space.sample()
            #     state, reward, done, _ = env.step(action)
            #
            #     if done or count > 1e2:
            #         break
            #     count += 1
            #
            # env.close()

if __name__ == '__main__':
    main()

