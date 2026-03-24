import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # 设置世界属性
        world.dim_c = 2
        world.collaborative = True  # 协作式任务
        # 添加智能体（单无人机）
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15  # 无人机尺寸
        # 添加地标（5个食物+3个障碍物）
        world.landmarks = [Landmark() for i in range(8)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'food_%d' % i if i < 5 else 'obstacle_%d' % (i - 5)
            landmark.collide = False
            landmark.movable = False
            # 适配缩小的尺寸
            if 'food' in landmark.name:
                landmark.size = 0.08  # 食物尺寸缩小
                landmark.color = np.array([0.1, 0.9, 0.1])  # 食物绿色
            else:
                landmark.size = 0.08  # 障碍物尺寸缩小（原0.1）
                landmark.color = np.array([0.9, 0.1, 0.1])  # 障碍物红色
        # 重置世界
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # 无人机初始位置（后续会被动力学强制重置到原点）
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # 随机设置食物位置（安全区范围内）
        for i, landmark in enumerate(world.landmarks):
            if 'food' in landmark.name:
                landmark.state.p_pos = np.random.uniform(-1.0, 1.0, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            # 障碍物固定位置
            elif 'obstacle' in landmark.name:
                if i == 5:
                    landmark.state.p_pos = np.array([0.5, 0.5])
                elif i == 6:
                    landmark.state.p_pos = np.array([-0.5, 0.5])
                else:
                    landmark.state.p_pos = np.array([0.0, -0.5])
                landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        """环境默认奖励（实际使用论文适配的自定义奖励）"""
        reward = 0.0
        # 碰撞障碍物惩罚（备用）
        if agent.collide:
            for landmark in world.landmarks:
                if 'obstacle' in landmark.name and self.is_collision(agent, landmark):
                    reward -= 10.0
        # 收集食物奖励（备用）
        for landmark in world.landmarks:
            if 'food' in landmark.name and self.is_collision(agent, landmark):
                reward += 20.0
        return reward

    def is_collision(self, agent, landmark):
        """碰撞检测（备用，使用缩小后的判定）"""
        delta_pos = agent.state.p_pos - landmark.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + landmark.size - 0.05  # 缩小判定阈值
        return True if dist < dist_min else False

    def observation(self, agent, world):
        """观测：自身速度+所有食物/障碍物相对位置"""
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        obs = np.concatenate([agent.state.p_vel] + entity_pos)
        return obs