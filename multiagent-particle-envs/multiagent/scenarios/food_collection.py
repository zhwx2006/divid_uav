import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # 设置世界属性
        world.dim_c = 2
        world.collaborative = True  # 协作式任务
        # 添加智能体
        world.agents = [Agent() for i in range(1)]  # 单无人机
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # 添加地标（食物+障碍物）
        world.landmarks = [Landmark() for i in range(8)]  # 5个食物+3个障碍物
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'food_%d' % i if i < 5 else 'obstacle_%d' % (i - 5)
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            if 'food' in landmark.name:
                landmark.color = np.array([0.1, 0.9, 0.1])  # 食物绿色
            else:
                landmark.color = np.array([0.9, 0.1, 0.1])  # 障碍物红色
        # 重置世界
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # 随机设置智能体位置
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # 随机设置食物位置
        for i, landmark in enumerate(world.landmarks):
            if 'food' in landmark.name:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
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

    # 核心修复：添加is_collision方法（检测智能体与地标碰撞）
    def is_collision(self, agent, landmark):
        delta_pos = agent.state.p_pos - landmark.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + landmark.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        reward = 0.0

        # 1. 碰撞惩罚（智能体与障碍物碰撞）- 缩放100倍：10 → 0.1
        if agent.collide:
            for landmark in world.landmarks:
                if 'obstacle' in landmark.name and self.is_collision(agent, landmark):
                    reward -= 0.1  # 碰撞障碍物惩罚（原10.0 → 0.1）

        # 2. 收集食物奖励（拆分大额奖励，降低方差）- 缩放100倍：200 → 2
        food_collected = 0
        for landmark in world.landmarks:
            if 'food' in landmark.name and self.is_collision(agent, landmark):
                food_collected += 1
                reward += 2.0  # 每个食物奖励（原200.0 → 2.0）

        # 3. 向最近食物移动的中间奖励 - 缩放100倍：0.5→0.005，0.1→0.001
        active_foods = [l for l in world.landmarks if 'food' in l.name]
        if active_foods:
            # 找到最近的食物
            closest_food = min(active_foods, key=lambda f: np.linalg.norm(agent.state.p_pos - f.state.p_pos))
            dist = np.linalg.norm(agent.state.p_pos - closest_food.state.p_pos)
            # 距离越小，奖励越高（最大0.005/步，原0.5/步）
            reward += max(0, 0.005 - 0.001 * dist)

        # 4. 靠近障碍物惩罚 - 缩放100倍：0.2 → 0.002
        for landmark in world.landmarks:
            if 'obstacle' in landmark.name:
                obs_dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                if obs_dist < 1.0:  # 距离小于1时开始惩罚
                    reward -= (1.0 - obs_dist) * 0.002  # 原0.2 → 0.002

        # 5. 保持移动奖励 - 缩放100倍：0.05 → 0.0005
        agent_speed = np.linalg.norm(agent.state.p_vel)
        if agent_speed > 0.01:
            reward += 0.0005  # 小幅奖励，鼓励移动（原0.05 → 0.0005）

        return reward
    def observation(self, agent, world):
        # 观测：自身位置+速度 + 所有食物/障碍物位置
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # 自身速度
        obs = np.concatenate([agent.state.p_vel] + entity_pos)
        return obs