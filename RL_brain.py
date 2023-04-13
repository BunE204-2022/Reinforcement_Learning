import numpy as np
import pandas as pd

# RL_brain是实现该游戏代码的核心，“brain”，这个代码主要是通过引入pandas中dataframe的结构储存状态-动作对(即Q表)，创建了RL类和Sarsa/Qlearning类(即值函数估计的公式实现)这两个大类

class RL(object):  # 个人理解：对应模拟了“采样”过程，将更新的状态一个一个地储存在q表中
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):  # 包含4个“参数”(元素),0.01实际就是α
        # __init__函数，一个特殊的函数，作用主要是事先把一些重要的属性填写进来；特点是第一个参数永远是self，表示创建的这个例子本身
        # 用actions lr gamma epsilon代表这些的“缩写”
        self.actions = action_space  # 动作维度，有几个动作可以选择
        self.lr = learning_rate  # 学习率，实际就是公式中的α
        self.gamma = reward_decay
        self.epsilon = e_greedy  # 按一定概率随机选动作

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 利用pandas中的数据框格式储存s-a对，列的名字为动作的名称，对象类型为np.float64

    def check_state_exist(self, state):  # 查询该状态是否存在在q表中
        if state not in self.q_table.index:
            # 如果状态在当前的Q表中不存在,将当前状态加入Q表中
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):  # 在某状态，选择动作
        self.check_state_exist(observation)
        # 从均匀分布的[0,1)中随机采样：当小于贪婪策略中epsilon时采用选择最优行为的方式；大于时选择随机行为的方式。这样人为增加随机性是为了解决陷入局部最优
        if np.random.rand() < self.epsilon:  # epsilon出现
            # 选择最优行为
            state_action = self.q_table.loc[observation, :]
            # 因为一个状态下最优行为可能会有多个,所以在碰到这种情况时,需要随机选择一个行为进行
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # # 选择随机行为
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):  # 跟下面的def learn？
        pass


# 异策略Q-learning
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):  # 学习率, 来决定这次的误差有多少是要被学习的；折扣因子为0.9；epsilon贪婪策略为0.9，说明有90% 的情况我会按照 Q 表的最优值选择行为, 10% 的时间使用随机选行为
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)  # super()是python中调用父类（超类）的一种方法

    def learn(self, s, a, r, s_):  # 少了一个a_
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]  # 在(与环境)交互前，其状态-动作对所对应的值函数Q
        if s_ != 'terminal':
            # 使用公式：Q_target = r+γ  maxQ(s',a')
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # TD目标，多了一个.max()
        else:
            q_target = r

        # 更新公式: Q(s,a)←Q(s,a)+α(r+γ  maxQ(s',a')-Q(s,a))
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


# 同策略Sarsa
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):  # 与Qlearning相比，多了一个a_
        self.check_state_exist(s_)  # 实现基于epsilon贪婪策略的最优动作选择，见上def choose_action
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # 使用公式: Q_taget = r+γQ(s',a')
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # Sarsa的TD目标，没有.max()
        else:
            q_target = r
        # 更新公式: Q(s,a)←Q(s,a)+α(r+γQ(s',a')-Q(s,a))
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 预测的Sarsa值函数
