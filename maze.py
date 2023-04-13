import numpy as np  # 导入numpy库并将其重命名为np。这样在使用numpy的函数时, 可以使用np.函数名的方式调用。用于科学计算、将数据转化为数组。
import sys  # python语言的一个系统内置模块，安装Python后自动包含sys库，不需要单独安装
import time  # 为了time.sleep()可加速/延迟，即用于延时

# maze是图像环境，即创建“网格世界”的一个环境

# 第8行代码是为看看python解释器版本号的主要部分是什么版本
if sys.version_info.major == 2:
    import Tkinter as tk  # Tkinter是python标准的GUI(Graphical User Interface,图形用户界面)库，提供了诸如创建窗口、创建组件、窗口大小和标题设置等等基础的功能。
else:
    import tkinter as tk  # 应该是python2就导入Tkinter，其他版本的导入tkinter（可能由于更新会有名称上的差异）

UNIT = 40  # 每个格子的边长，正方形
MAZE_H = 5  # 行数
MAZE_W = 5  # 列数


class Maze(tk.Tk, object):
    def __init__(self):  # __init__函数，一个特殊的函数，作用主要是事先把一些重要的属性填写进来；特点是第一个参数永远是self，表示创建的这个例子本身
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # 动作空间：上下左右
        self.nS = np.prod([MAZE_H, MAZE_W])  # np.prod表示行与列的乘积，应该是构造空间的大小？
        self.n_actions = len(self.action_space)  # 动作空间的“长度”
        self.title('最优迷宫路线')  # 画面的标题
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))  # ？？
        self._build_maze()  # 见下

    def _build_maze(self):
        # 创建一个画布
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)  # 画布背景为白色；高=行数*格子大小，宽=列数*格子大小

        # 在画布上画出列
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT  # (x0,y0)表示线的“起点”坐标，(x1,y1)表示线的“终点”坐标
            self.canvas.create_line(x0, y0, x1, y1)  # 实际是在tk.Canvas()中添加一些线
        # 在画布上画出行
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 创建探险者起始位置(默认为左上角)
        origin = np.array([20, 20])  # 原点应该在左上角，而不是左下角

        # 陷阱1
        hell1_center = origin + np.array([UNIT * 3, 0])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 20, hell1_center[1] - 20,
            hell1_center[0] + 20, hell1_center[1] + 20,
            fill='black')
        # 陷阱2
        hell2_center = origin + np.array([0, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 20, hell2_center[1] - 20,
            hell2_center[0] + 20, hell2_center[1] + 20,
            fill='black')

        # 陷阱3
        hell3_center = origin + np.array([UNIT * 3, UNIT])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 20, hell3_center[1] - 20,
            hell3_center[0] + 20, hell3_center[1] + 20,
            fill='black')

        # 陷阱4
        hell4_center = origin + np.array([UNIT, UNIT * 2])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 20, hell4_center[1] - 20,
            hell4_center[0] + 20, hell4_center[1] + 20,
            fill='black')

        # 陷阱5
        hell5_center = origin + np.array([UNIT * 2, UNIT * 4])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 20, hell5_center[1] - 20,
            hell5_center[0] + 20, hell5_center[1] + 20,
            fill='black')

        # 陷阱6
        hell6_center = origin + np.array([UNIT * 3, UNIT * 4])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 20, hell6_center[1] - 20,
            hell6_center[0] + 20, hell6_center[1] + 20,
            fill='black')

        # 陷阱7
        hell7_center = origin + np.array([UNIT * 4, UNIT * 4])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 20, hell7_center[1] - 20,
            hell7_center[0] + 20, hell7_center[1] + 20,
            fill='black')

        # ”出口“位置
        self.oval = self.canvas.create_text(180,100,text='出口',font=("Purisa", 15))  # (180,100)表示坐标中心

        # 将探险者用矩形表示
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='green')

        # 画布展示
        self.canvas.pack()

    # 根据当前的状态重置画布(为了展示动态效果)
    def reset(self):
        self.update()
        time.sleep(0.5)  # 暂停0.5秒后再更新，所以死了马上重新开始，比较快
        self.canvas.delete(self.rect)  # 删除原有代表探险者的长方形
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='green')  # 重新构建一个探险者，并从初始位置开始？
        return self.canvas.coords(self.rect)  # canvas的coords方法用于重新设置坐标

    # 根据当前行为,确认下一步的位置。我的理解：为在画布上下左右行动做定义，比如往上走就是纵轴-40，下就是纵轴+40，左右是横轴改变，同理。
    def step(self, action):  # step是物理引擎，用来输入动作
        s = self.canvas.coords(self.rect)  # canvas的coords方法用于重新设置坐标
        base_action = np.array([0, 0])
        if action == 0:  # 上
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 下
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 左
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # 右
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        # 在画布上将探险者移动到下一位置
        self.canvas.move(self.rect, base_action[0], base_action[1])
        # 重新渲染整个界面
        s_ = self.canvas.coords(self.rect)
        oval_flag = False

        # 根据当前位置来获得回报值,及是否终止
        if s_ == self.canvas.coords(self.oval):  # 若当前处在出口处，则回报为1，终止状态
            reward = 1
            done = True
            s_ = 'terminal'
            oval_flag = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4), self.canvas.coords(self.hell5), self.canvas.coords(self.hell6),
                    self.canvas.coords(self.hell7)]:  # 若探险者在陷阱中，则回报为-1，也会终止本次探险（开始新的探险）
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0  # 在其他格子中，回报为0，探险继续，不会终止程序
            done = False

        return s_, reward, done, oval_flag

    def render(self):  # 图像引擎
        time.sleep(0.1)  # 探险者走的比较快，若设为1秒，走的就会比较慢
        self.update()

    # 根据传入策略进行界面的渲染。【应该就是与智能体的“联系”？我猜……】
    def render_by_policy(self, policy):
        cal_policy = sorted(policy)

        pre_x, pre_y = 20, 20

        for state in cal_policy:
            x = (state[0] + state[2]) / 2
            y = (state[1] + state[3]) / 2

            self.canvas.create_line(pre_x, pre_y, x, y, fill="green", tags="line", width=5)

            pre_x = x
            pre_y = y

        # 连接到出口位置
        oval_center = [20, 20] + np.array([UNIT * 2, UNIT * 4])

        self.canvas.create_line(pre_x, pre_y, oval_center[0], oval_center[1], fill="green", tags="line", width=5)

        self.render()  # 最后进行渲染

    def render_by_policy_new(self, policy):
        for i in range(MAZE_W):
            rows_obj = policy[i]
            for j in range(MAZE_H):
                item_center_x, item_center_y = (j * UNIT + UNIT / 2), (i * UNIT + UNIT / 2)

                cols_obj = rows_obj[j]

                if cols_obj == -1:
                    continue

                for item in cols_obj:
                    if item == 0:  # 上
                        item_x = item_center_x
                        item_y = item_center_y - 15.0
                        self.canvas.create_line(item_center_x, item_center_y, item_x, item_y, fill="black", width=1,
                                                arrow='last')
                    elif item == 1:  # 下
                        item_x = item_center_x
                        item_y = item_center_y + 15.0
                        self.canvas.create_line(item_center_x, item_center_y, item_x, item_y, fill="black", width=1,
                                                arrow='last')
                    elif item == 2:  # 左
                        item_x = item_center_x - 15.0
                        item_y = item_center_y
                        self.canvas.create_line(item_center_x, item_center_y, item_x, item_y, fill="black", width=1,
                                                arrow='last')
                    elif item == 3:  # 右
                        item_x = item_center_x + 15.0
                        item_y = item_center_y
                        self.canvas.create_line(item_center_x, item_center_y, item_x, item_y, fill="black", width=1,
                                                arrow='last')
        self.render()  # 最后渲染
