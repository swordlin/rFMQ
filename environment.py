
import matplotlib.pyplot as plt

# 画图
import pandas as pd

# class climbingGame:
#     def __init__(self):
#         self.agent = 2
#         self.action = ['a','b','c']
#         self.reward = pd.DataFrame([[11,-30,0],[-30,7,6],[0,0,5]],\
#                                    columns=['a','b','c'],index=['a','b','c'])
#     def returnReward(self,a1,a2):
#         return self.reward.loc[a1,a2]

import numpy as np
import copy
import pickle


class RFMQ(object):

	def __init__(self, epsilon, n_actions, n_agents, alpha, alpha_f):
		self.epsilon = epsilon
		self.n_actions = n_actions
		self.n_agents = n_agents
		self.alpha = alpha  # 用于Q值更新的新知识学习率
		self.alpha_f = alpha_f  # 用于F值更新的历史最大回报学习率

		self.A = np.zeros(shape=(n_agents, n_actions))  # n_agents个智能体的动作空间, A(i,a)表示第i个智能体第a个动作代表的值
		self.Q = np.zeros(shape=(n_agents, n_actions))  # 每个智能体各自维护一个Q表, Q(i,a)表示第i个智能体选择动作a的总回报。
		self.Q_max = np.zeros(shape=(n_agents, n_actions))  # 每个智能体对应每个动作的历史最大回报值
		self.F = np.zeros(shape=(n_agents, n_actions))  # 每个智能体各自维护一个频率F表，F(i,a)表示第i个智能体选择过a的频率
		self.E = np.zeros(shape=(n_agents, n_actions))  # E(i,a)表示第i个智能体选择动作a的加权回报，这个似乎不需要记录吧，因为没有历史E值这个说法
		self.pi = np.zeros(shape=(n_agents, n_actions))  # 表示策略，pi(i,a)表示第i个智能体选择动作a的概率？咦，这不应该是确定性策略么。。
		# self.Q_old = np.zeros(shape=(n_agents, n_actions))      # 存放旧Q值，上面的self.Q存放每步更新的新Q值，但是E在计算的时候还使用旧Q值，旧Q值经过一定步数才更新, todo:这个不属于纯RFMQ。

		# self.action = ['a', 'b', 'c']
		# self.reward = pd.DataFrame([[11, -30, 0], [-30, 7, 6], [0, 0, 5]], columns=['a', 'b', 'c'], index=['a', 'b', 'c'])
		self.reward = np.array([
			[11, -30, 0],
			[-30, 7, 6],
			[0, 0, 5],
		])
	def choose_action(self):
		'''
		按照epsilon-greedy策略为所有智能体选择动作(的索引)。
		attention，这里的动作一律指的是索引，是指定动作在self.A中的索引
		:return: 
		'''
		joint_actions = np.zeros(shape=(self.n_agents), dtype=int)  # 由各个智能体选择动作组成的联合动作，实质上是动作的索引
		for i in range(self.n_agents):
			if np.random.random() < self.epsilon:
				# 探索,从所有动作中按照均匀分布随机选择一个动作(的索引)
				action = np.random.randint(0, len(self.A[i]))  # 随机选择第index个索引
				# action = self.A[i][index]  # 取出第I号智能体
			else:
				# 利用
				action = np.argmax(self.pi[i])  # 返回最大的pi值对应的索引。如果相同则返回第一个？
			joint_actions[i] = action
		return joint_actions

	def update_Q_value(self, joint_actions, reward):
		'''
		根据即时回报和历史Q值来更新当前Q表
		:param reward: 本步联合动作带来的即时奖励
		:return: 
		'''
		for i in range(self.n_agents):  # 为每个智能体更新自己的Q表
			a = joint_actions[i]  # 取出第i个智能体对应的动作索引
			# print(a)
			self.Q[i][a] = (1 - self.alpha) * self.Q[i][a] + self.alpha * reward
			if self.Q[i][a] > self.Q_max[i][a]:  #
				self.Q_max[i][a] = self.Q[i][a]

	def update_Q_old_value(self):
		'''
		经过一定步数之后调用本方法，使用当前的Q值更新掉旧的Q_old
		:return: 
		'''
		self.Q_old = copy.deepcopy(self.Q)

	def update_F_value(self, joint_actions, reward):
		'''
		根据即时回报、历史Q值、历史最大Q值来更新频率F值
		:param joint_actions: 
		:param reward: 
		:return: 
		'''
		for i in range(self.n_agents):  # 为每个智能体更新自己的频率
			a = joint_actions[i]  # 取出第i个智能体对应的动作索引
			if reward > self.Q_max[i][a]:
				self.Q_max[i][a] = reward
				self.F[i][a] = 1
			elif reward == self.Q_max[i][a]:
				self.F[i][a] = (1 - self.alpha_f) * self.F[i][a] + self.alpha_f
			else:
				self.F[i][a] = (1 - self.alpha_f) * self.F[i][a]

	def update_E_value(self, joint_actions):
		for i in range(self.n_agents):  # 为每个智能体更新自己的加权回报
			a = joint_actions[i]
			self.E[i][a] = (1 - self.F[i][a]) * self.Q[i][a] + self.F[i][a] * self.Q_max[i][a]
			# self.E[i][a] = (1 - self.F[i][a]) * self.Q_old[i][a] + self.F[i][a] * self.Q_max[i][a]

	def update_policy(self):
		'''
		更新策略pi
		:return: 
		'''
		for i in range(self.n_agents):
			if self.E[i][np.argmax(self.pi[i])] != np.max(self.E[i]):  # 对一个智能体而言，如果最大策略对应的加权回报，不等于最大加权回报
				# 首先找出最大E值对应的所有动作集合
				# print(self.E[i])
				E_max = np.max(self.E[i])
				# print(E_max)
				max_actions = []
				for j in range(len(self.E[i])):
					if self.E[i][j] == E_max:
						max_actions.append(j)  # 找到最大E值对应的所有动作(的索引)，不知道有没有更简便的写法
				# print(max_actions)
				max_action = np.random.choice(max_actions)  # 从中随机挑选一个动作,

				self.pi[i] = np.zeros(shape=(self.n_actions))  # 更新策略，事实上，我不太确定这块存在的意义
				self.pi[i][max_action] = 1

	def observe_immediate_reward(self, a1, a2):
		print(a1, a2)
		return self.reward[a1,a2]

	# def save_model(self):
	# 	'''
	# 	负责保存模型，直接实例化整个模型到本地，用于程序中断时的现场恢复
	# 	:return:
	# 	'''
	# 	with open(PATH.model_save_path, 'wb') as writer:
	# 		pickle.dump(self, writer)

def main():
	n_actions = 3  # 所有智能体的动作空间长度是一样的
	n_agents = 2  # 智能体的个数
	epsilon = 0.1      # 探索比例
	alpha = 0.1  # 新知识学习率
	alpha_f = 0.05  # 频率更新的学习率, 后三个参数来源于rFMQ原始论文

	rFMQ = RFMQ(epsilon, n_actions, n_agents, alpha, alpha_f)

	n_episodes = 1     # 训练重复的回合数
	n_steps = 10000       # 每个回合需要执行的步数，TODO：这里回合这个概念没有意义，因为这个游戏实际上没有所谓的回合终止条件，无所谓成败。
	rewards = np.zeros(shape=(n_episodes, n_steps))  # 用来存放历史rewards表，即历史正确率表，用来观察结果
	for episode in range(n_episodes):  # 训练10个回合
		# 初始化参数，initialize parameters
		print('episode:{}'.format(episode))
		# print(rFMQ.E)
		for step in range(n_steps):  # 每个回合走10步
			# 第三层循环，针对智能体
			# 基于\epsilon-greedy为所有智能体选择动作，选择策略为，探索和利用，"利用"变成了从动作空间选择具有最高E值的动作，就是说Q值更新并记录，但是E值不进行更新。
			# 针对每个动作，基于其历史最大回报，更新每个动作对应的频率F(这个历史最大回报应该不包括刚选择的动作吧。)
			# 为选择出的每个动作更新总回报Q

			# 首先根据策略和epsilon贪婪为10个智能体分别选择动作
			# 应用联合动作，观察即时回报r
			# 更新Q值
			# 更新F值
			# 计算E值
			# 更新策略
			# 选择联合动作
			joint_actions = rFMQ.choose_action()
			# 应用联合动作，观察即时回报r
			reward = rFMQ.observe_immediate_reward(*joint_actions)
			rewards[episode][step] = reward
			print('当前第{}步,回报为{}'.format(step, reward))
			# 更新Q值
			rFMQ.update_Q_value(joint_actions, reward)
			# 更新频率F值
			rFMQ.update_F_value(joint_actions, reward)
			# 计算加权回报值
			rFMQ.update_E_value(joint_actions)
			# 更新策略
			rFMQ.update_policy()
	print(rFMQ.E)
if __name__ == '__main__':
	main()

