import logging
import numpy
import random
from gym import spaces
import gym


class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        #状态空间
        self.states = [1, 2, 3, 4, 5, 6, 7, 8] #状态空间对应8个格子
        self.observation_space = spaces.Discrete(8)#discrete是一种类，数字几就是几类

        #位置
        self.x = [140, 220, 300, 380, 460, 140, 300, 460]#对应每一个格子的中心，其实是后来机器人位置的确定
        self.y = [250, 250, 250, 250, 250, 150, 150, 150]
        self.terminate_states = dict()  #终止状态为字典格式
        self.terminate_states[6] = 1#反正1就都是终止，要么死了要么找到金币了
        self.terminate_states[7] = 1
        self.terminate_states[8] = 1

        #动作空间
        self.actions = ['n','e','s','w']#动作空间，往东南西北走
        self.action_space = spaces.Discrete(4)

        #回报函数
        self.rewards = dict()        #回报的数据结构为字典
        self.rewards['1_s'] = -1.0   #在1往南走就死了，下同
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        #状态转移概率
        self.t = dict()             #状态转移的数据格式为字典，从某格往哪走就到了哪里
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 8
        self.t['5_w'] = 4

        self.gamma = 0.8         #折扣因子
        self.viewer = None       #显示器
        self.state = None        #状态


    def transform(self,state,action):
        # 遍历动作空间，当不在状态转移概率中时，该状态设为-1
        s = -1  #有些方向走不通
        r = 0   #回报
        key = '%i_%s' % (state, action)
        if key in self.rewards:   #判断是否是终止状态
            r = self.rewards[key]
        if key in self.t:
            s = self.t[key]       #那么状态s就要改变了
        return self.t, s, r


    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self, s):
        self.state = s

    def step(self, action):  # step函数输入的是动作，输出的是下一个时刻的动作、回报、是否终止和调试信息
        # 系统当前状态
        state = self.state
        if state in self.terminate_states:  # 判断系统当前是否处于终止状态
            return state, 0, True, {}  # 调试信息可以为空，但是不能缺少，否则会报错，常用{}表示
        key = "%d_%s" % (state, action)  # 将状态和动作组成字典的键值

        # 状态转移
        if key in self.t:
            next_state = self.t[key]  # 如果是可以走的路，这里就是下一个状态
        else:
            next_state = state  # 相当于原地打转
        self.state = next_state  # 状态进行重新赋值
        is_terminal = False  # 先给一个还没到终止位置的初值

        if next_state in self.terminate_states:  # 判断是否到了终止位置
            is_terminal = True

        if key not in self.rewards:  # 只有那三步会有回报，其他都是0
            r = 0.0
        else:
            r = self.rewards[key]

        return next_state, r, is_terminal, {}

    def reset(self):  # 一开始随机产生一个位置
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 600  # 整个坐标系的横轴与纵轴
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)  # 规定长和宽

            # 创建网格世界
            self.line1 = rendering.Line((100, 300), (500, 300))  # 一共是11条线，用坐标轴来绘制
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (100, 100))
            self.line4 = rendering.Line((180, 300), (180, 100))
            self.line5 = rendering.Line((260, 300), (260, 100))
            self.line6 = rendering.Line((340, 300), (340, 100))
            self.line7 = rendering.Line((420, 300), (420, 100))
            self.line8 = rendering.Line((500, 300), (500, 100))
            self.line9 = rendering.Line((100, 100), (180, 100))
            self.line10 = rendering.Line((260, 100), (340, 100))
            self.line11 = rendering.Line((420, 100), (500, 100))

            # 创建第一个骷髅
            self.kulo1 = rendering.make_circle(40)  # 画一个直径为40的圆
            self.circletrans = rendering.Transform(translation=(140, 150))  # 规定了圆心的位置
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(0, 0, 0)  # 设置为黑色

            # 创建第二个骷髅
            self.kulo2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(460, 150))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(0, 0, 0)

            # 创建金币
            self.gold = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(300, 150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)  # 颜色不一样

            # 创建机器人
            self.robot = rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            # 让这些线也变成黑色的
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None: return None
        # self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        self.robotrans.set_translation(self.x[self.state - 1], self.y[self.state - 1])  # 没太明白，为什么是要减1的

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

