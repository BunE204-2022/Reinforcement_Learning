# main这一代码其实包括3个模块：def get_action、def get_policy及def update()

import sys  # python语言的一个系统内置模块，安装Python后自动包含sys库，不需要单独安装

if "../" not in sys.path:
    sys.path.append("../")  # 把路径添加到系统的环境变量
from maze import Maze  # 导入环境
from RL_brain import QLearningTable, SarsaTable  # 导入“策略”
import numpy as np  # 导入numpy库并将其重命名为np。这样在使用numpy的函数时, 可以使用np.函数名的方式调用。用于科学计算、将数据转化为数组。

METHOD = "SARSA"
#METHOD = "Q-Learning"


def get_action(q_table, state):  # 如何从q表中选最优动作？
    # 选择最优动作
    state_action = q_table.loc[state, :]  # 将q表中状态state下的动作对赋值给state_action
    # 因为一个状态下最优动作可能会有多个,所以在碰到这种情况时,需要随机选择一个动作
    state_action_max = state_action.max()

    idxs = []  # 创建一个空列表(一会儿用于储存动作？)

    for max_item in range(len(state_action)):  # max_item是一个临时变量，实际就是遍历所有的state_action(动作)
        if state_action[max_item] == state_action_max:
            idxs.append(max_item)  # 若其为最优动作，则将该动作扩展添加到idxs中

    sorted(idxs)  # 给这些动作进行排序
    return tuple(idxs)  # 将对象转化为元组，并返回


def get_policy(q_table, rows=5, cols=5, pixels=40, orign=20):  # 定义策略
    policy = []

    for i in range(rows):  # i对应每行
        for j in range(cols):  # j对应每列
            # 求出每个各自的状态
            item_center_x, item_center_y = (j * pixels + orign), (i * pixels + orign)  # 状态的“中心”
            item_state = [item_center_x - 15.0, item_center_y - 15.0, item_center_x + 15.0, item_center_y + 15.0]  # 用一整个长方形块来表示对应状态

            # 如果当前状态为各终止状态,则值为-1
            if item_state in [env.canvas.coords(env.hell1), env.canvas.coords(env.hell2),
                              env.canvas.coords(env.hell3), env.canvas.coords(env.hell4),
                              env.canvas.coords(env.hell5), env.canvas.coords(env.hell6),
                              env.canvas.coords(env.hell7), env.canvas.coords(env.oval)]:
                policy.append(-1)
                continue

            if str(item_state) not in q_table.index:
                policy.append((0, 1, 2, 3))
                continue

            # 选择最优动作
            item_action_max = get_action(q_table, str(item_state))

            policy.append(item_action_max)

    return policy


def update():  # 更新环境
    # 游戏运行次数
    for episode in range(100):
        # 重新初始化状态(即observation)
        observation = env.reset()

        c = 0

        tmp_policy = {}

        while True:
            # 渲染当前环境=更新界面
            env.render()

            # 基于当前状态(observation)选择行为a
            action = RL.choose_action(str(observation))

            state_item = tuple(observation)

            tmp_policy[state_item] = action

            # 采取该行为，并获得下一个状态和回报,及是否终止(flag)
            observation_, reward, done, oval_flag = env.step(action)

            if METHOD == "SARSA":
                # 基于下一个状态选择行为
                action_ = RL.choose_action(str(observation_))

                # 基于变化 (s, a, r, s', a)使用Sarsa进行Q表的更新
                RL.learn(str(observation), action, reward, str(observation_), action_)  # 因为行动a和评估策略是同策略

            elif METHOD == "Q-Learning":
                # 根据当前的变化开始更新Q
                RL.learn(str(observation), action, reward, str(observation_))

            # 改变状态和行为，并进入狭义循环(?)
            observation = observation_  # 赋值为下一个状态
            c += 1
            # 如果为终止状态,结束当前的局数
            if done:
                break

    print('游戏结束')
    # 开始输出最终的Q表
    q_table_result = RL.q_table
    # 使用Q表输出各状态的最优策略
    policy = get_policy(q_table_result)
    print("最优策略为", end=":")
    print(policy)
    print("迷宫格式为", end=":")
    policy_result = np.array(policy).reshape(5, 5)
    print(policy_result)
    print("根据求出的最优策略画出方向")
    env.render_by_policy_new(policy_result)
    print(item_action_max)

    #env.destroy()  # 关闭游戏窗口


if __name__ == "__main__":  # 只有在main才能打开
    env = Maze()  # 建造游戏环境
    RL = SarsaTable(actions=list(range(env.n_actions)))
    if METHOD == "Q-Learning":
        RL = QLearningTable(actions=list(range(env.n_actions)))  # 使用0-3代表迷宫游戏的四个动作
    # 开始可视化环境 env
    env.after(100, update)  # 100?
    env.mainloop()
