import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        world.collaborative = True

        # 核心配置
        world.boundary = (-1.0, 1.0)
        world.collision_terminate = False
        self.collision_radius = 0.0  # 取消无人机碰撞半径，仅本体触发碰撞

        # 无人机配置
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = self.collision_radius  # 无人机半径设为0
            agent.max_speed = 0.1

        # 食物/障碍物半径配置
        obj_radius_scale = 0.6
        world.landmarks = [Landmark() for i in range(8)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'food_%d' % i if i < 5 else 'obstacle_%d' % (i - 5)
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.collision_radius * obj_radius_scale if self.collision_radius > 0 else 0.1
            if 'food' in landmark.name:
                landmark.color = np.array([0.1, 0.9, 0.1])
            else:
                landmark.color = np.array([0.9, 0.1, 0.1])

        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.collision_terminate = False
        # 无人机初始位置
        for agent in world.agents:
            agent.state.p_pos = np.clip(
                np.random.uniform(-0.8, +0.8, world.dim_p),
                world.boundary[0] + 0.1,
                world.boundary[1] - 0.1
            )
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # 食物/障碍物位置
        for i, landmark in enumerate(world.landmarks):
            if 'food' in landmark.name:
                landmark.state.p_pos = np.clip(
                    np.random.uniform(-0.9, +0.9, world.dim_p),
                    world.boundary[0] + landmark.size,
                    world.boundary[1] - landmark.size
                )
            elif 'obstacle' in landmark.name:
                if i == 5: landmark.state.p_pos = np.array([0.4, 0.4])
                elif i == 6: landmark.state.p_pos = np.array([-0.4, 0.4])
                else: landmark.state.p_pos = np.array([0.0, -0.4])
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent, entity):
        """仅当无人机中心点碰到物体本体时触发碰撞"""
        delta_pos = agent.state.p_pos - entity.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        collide_dist = entity.size
        return dist <= collide_dist

    def is_boundary_collision(self, agent, world):
        """仅当无人机中心点超出边界时触发"""
        min_b, max_b = world.boundary
        x_out = agent.state.p_pos[0] < min_b or agent.state.p_pos[0] > max_b
        y_out = agent.state.p_pos[1] < min_b or agent.state.p_pos[1] > max_b
        return x_out or y_out

    def reward(self, agent, world):
        reward = 0.0  # 初始奖励为0，无存活奖励

        # 1. 边界碰撞惩罚
        if self.is_boundary_collision(agent, world):
            reward -= 50.0  # 仅惩罚，无其他奖励
            world.collision_terminate = True
            return reward

        # 2. 障碍物碰撞惩罚
        for landmark in world.landmarks:
            if 'obstacle' in landmark.name and self.is_collision(agent, landmark):
                reward -= 30.0  # 仅惩罚，无其他奖励
                world.collision_terminate = True
                return reward

        # 3. 仅保留食物收集奖励（无任何探索/靠近奖励）
        for landmark in world.landmarks:
            if 'food' in landmark.name and self.is_collision(agent, landmark):
                reward += 20.0  # 仅收集食物时奖励
                # 食物重置位置（远离障碍物）
                while True:
                    new_pos = np.clip(
                        np.random.uniform(-0.9, +0.9, world.dim_p),
                        world.boundary[0] + landmark.size,
                        world.boundary[1] - landmark.size
                    )
                    # 确保新位置远离障碍物
                    obstacle_dist = [np.sqrt(np.sum(np.square(new_pos - obs.state.p_pos)))
                                     for obs in world.landmarks if 'obstacle' in obs.name]
                    if all(d > 0.3 for d in obstacle_dist):
                        landmark.state.p_pos = new_pos
                        break

        # 完全移除：靠近食物/远离障碍物的引导奖励（仅保留收集食物奖励+碰撞惩罚）
        return reward

    # 观测函数
    def observation(self, agent, world):
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        # 仅保留基础观测（移除距离特征，避免引导）
        vel = agent.state.p_vel
        return np.concatenate([vel] + entity_pos)

    # 兼容任意参数的done方法
    def done(self, *args):
        if len(args) >= 2:
            agent = args[0]
            world = args[1]
            agent.state.p_pos = np.clip(
                agent.state.p_pos,
                world.boundary[0],
                world.boundary[1]
            )
            return world.collision_terminate
        return False