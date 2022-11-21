#1.1 2处报错

import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()

#%%

#1.2 一闪而过

import gym
env = gym.make('CartPole-v1' ,render_mode='human')
env.reset()
env.render()

#2.1 2处报错
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

#%%

#2.2 一直运行，不停止：必须加上env.close()
import gym
env = gym.make('CartPole-v1' ,render_mode='human') #调用函数创建环境对象，每个环境都有一个ID，环境名称最后表示版本号。'human'表示在人类显示器或终端上渲染，才可输出图像。
#智能体通过调用env类的方法来与环境进行交互，常用的env类方法有：reset(), render(), step(), close()等等。
env.reset()
for _ in range(1000):
   env.render()
   env.step(env.action_space.sample())

#%%

#3 正常运行
import gym
def main():
    env = gym.make('CartPole-v1', render_mode = 'human')
    for i_episode in range(20): #episode为每一次尝试
        observation = env.reset() #环境重置，一次尝试结束，智能体需要从头开始，即需要具有重新初始化的功能
        for t in range(100):
            env.render() #图像引擎
            print(observation)
            action = env.action_space.sample() #动作采样
            observation, reward, done, info, _ = env.step(action) #单步交互
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    env.close()
main()

#%%

#4.慢版
import gym
import time

# 生成环境
env = gym.make('CartPole-v1', render_mode='human')#总算找到了一个能够正确地进行rendering的方式，在调用make创建env时指定render_mode参数，然后，不需要再调用env.render()
# 环境初始化
state = env.reset()
# 循环交互

while True:
    # 渲染画面
    # env.render()
    # 从动作空间随机获取一个动作
    action = env.action_space.sample()
    # agent与环境进行一步交互
    state, reward, terminated, truncated, info = env.step(action)#注：新版本的env.step()的返回值由4个变为5个了，done修改扩展为terminated,truncated
    print('state = {0}; reward = {1}'.format(state, reward))
    # 判断当前episode 是否完成
    if terminated:
        print('terminated')
        break
    time.sleep(0.1)#便于以慢动作的方式放映整个过程
# 环境结束
# env.close()

#%%

#5.源代码
import logging #用来日志
import math #有很多函数
import gym
from gym import spaces #gym.spaces()定义状态和动作空间
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__) #logging包要把设置读取到实例中才能用

class CartPoleEnv(gym.Env): #创建类别
    metadata = { #元数据，用于支持可视化的一些设定，改变渲染环境时的参数
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self): #初始化
        self.gravity = 9.8                                   #重力加速度
        self.masscart = 1.0                                  #小车的质量
        self.masspole = 0.1                                  #摆的质量
        self.total_mass = (self.masspole + self.masscart)    #总质量=小车的质量+摆的质量
        self.length = 0.5 # actually half the pole's length  #摆的半长，即摆的转动中心到摆的质心的距离
        self.polemass_length = (self.masspole * self.length) #摆的质量长=摆的质量x摆的半长
        self.force_mag = 10.0                                #外力的振幅
        self.tau = 0.02  # seconds between state updates     #更新时间步

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360 #摆角最大范围
        self.x_threshold = 2.4                                #小车x方向最大运动范围

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)                      #小车倒立摆的动作空间为2维离散空间
        self.observation_space = spaces.Box(-high, high)            #观测空间为（x,dx,theta,dtheta）区间为（（-24，24），（-2.4，2.4））

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        #初始化定义结束

    def _seed(self, seed=None): #指定环境的随机数种子，但seed()函数在新版本好像已经被删除了，这是作者在github上的笔记。新版本中，seed应该在reset()函数调用时指定
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action): #用于编写智能体与环境交互的逻辑；它接受一个动作（action）的输入，根据action给出下一时刻的状态（state）、当前动作的回报（reward）、探索是否结束（done）及调试帮助信息信息。
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state #系统当前状态
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta) #余弦函数
        sintheta = math.sin(theta) #正弦函数
        #车摆的动力学方程式，即加速度与动作之间的关系
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass #小车的平移加速度
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc #积分求下一步的状态
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done) #True or False

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self): #重新初始化函数：每轮开始前重置智能体的状态
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))# 利用均匀随机分布初始化环境的状态
        self.steps_beyond_done = None #设置当前步数为None
        return np.array(self.state) #返回环境的初始化状态

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering #导入rendering模块，利用其中的画图函数进行图形绘制
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # 创建小车的代码
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) #填充一个矩形
            #添加台车转换矩阵属性
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            #加入几何体台车
            self.viewer.add_geom(cart)
            #创建摆杆
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            #添加摆杆转换矩阵属性
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            #加入几何体
            self.viewer.add_geom(pole)
            #创建摆杆和台车之间的连接
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            #创建台车来回滑动的轨道，即一条直线
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        #设置平移属性
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')