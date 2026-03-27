# -*- coding: utf-8 -*-
# 开发者：Bright Fang
# 开发时间：2023/7/20 23:30
import numpy as np
import copy
import gymnasium as gym
from assignment import constants as C
from gymnasium import spaces
import math
import random
import pygame
from assignment.components import player
from assignment import tools
from assignment.components import info


class RlGame(gym.Env):
    def __init__(self, n, m, render=False):
        super().__init__()
        self.hero_num = n
        self.enemy_num = m

        # ===================== 你要的修改 =====================
        self.obstacle_num = 1    # 障碍物 = 1
        self.goal_num = 5        # 终点 = 5
        # ======================================================

        self.Render = render
        self.game_info = {
            'epsoide': 0,
            'hero_win': 0,
            'enemy_win': 0,
            'win': '未知',
        }

        self.MIN_X = C.ENEMY_AREA_X + 50
        self.MAX_X = C.ENEMY_AREA_WITH - 50
        self.MIN_Y = C.ENEMY_AREA_Y + 50
        self.MAX_Y = C.ENEMY_AREA_HEIGHT - 50
        self.OBSTACLE_SAFE_RADIUS = 50
        self.EDGE_PENALTY = -1
        self.reached_goal_num = 0
        self.goal_exists = [True] * self.goal_num
        self.done_reason = 'other'

        if self.Render:
            pygame.init()
            pygame.mixer.init()
            self.SCREEN = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
            pygame.display.set_caption("基于深度强化学习的空战场景无人机路径规划软件")
            self.GRAPHICS = tools.load_graphics(
                'C:/Users/22895/Desktop/UAV-path-planning-main - 修改版/Multi-UAVs path planning/path planning/assignment/source/image')
            self.SOUND = tools.load_sound(
                'C:/Users/22895/Desktop/UAV-path-planning-main - 修改版/Multi-UAVs path planning/path planning/assignment/source/music')
            self.clock = pygame.time.Clock()
            pygame.time.set_timer(C.CREATE_ENEMY_EVENT, C.ENEMY_MAKE_TIME)

        low = np.array([-1, -1])
        high = np.array([1, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def generate_random_obstacles(self):
        obstacles = []
        for _ in range(self.obstacle_num):
            x = random.randint(self.MIN_X, self.MAX_X)
            y = random.randint(self.MIN_Y, self.MAX_Y)
            obstacles.append((x, y))
        self.obstacle_positions = obstacles

    def generate_random_goals(self):
        goals = []
        # 确保终点之间至少间隔 80 像素，不会重叠
        min_goal_distance = 80
        for _ in range(self.goal_num):
            for attempt in range(200):
                x = random.randint(self.MIN_X, self.MAX_X)
                y = random.randint(self.MIN_Y, self.MAX_Y)
                # 先检查离障碍物远
                too_close_obstacle = False
                for (ox, oy) in self.obstacle_positions:
                    if math.hypot(x - ox, y - oy) < self.OBSTACLE_SAFE_RADIUS:
                        too_close_obstacle = True
                        break
                if too_close_obstacle:
                    continue

                # 再检查离其他终点远
                too_close_goal = False
                for (gx, gy) in goals:
                    if math.hypot(x - gx, y - gy) < min_goal_distance:
                        too_close_goal = True
                        break
                if not too_close_goal:
                    goals.append((x, y))
                    break
        return goals

    def start(self):
        self.finished = False
        self.set_battle_background()
        self.set_enemy_image()
        self.set_hero_image()
        self.generate_random_obstacles()
        self.set_obstacle_image()
        self.goal_positions = self.generate_random_goals()
        self.set_goal_image()

        self.info = info.Info('battle_screen', self.game_info)
        self.trajectory_x, self.trajectory_y = [], []
        self.enemy_trajectory_x, self.enemy_trajectory_y = [[] for _ in range(self.enemy_num)], [[] for _ in range(self.enemy_num)]
        self.reached_goal_num = 0
        self.goal_exists = [True] * self.goal_num
        self.done_reason = 'other'

    def set_battle_background(self):
        self.battle_background = self.GRAPHICS['background']
        self.battle_background = pygame.transform.scale(self.battle_background, C.SCREEN_SIZE)
        self.view = self.SCREEN.get_rect()

    def set_hero_image(self):
        self.hero = self.__dict__
        self.hero_group = pygame.sprite.Group()
        self.hero_image = self.GRAPHICS['fighter-blue']
        for i in range(self.hero_num):
            self.hero['hero' + str(i)] = player.Hero(image=self.hero_image)
            self.hero_group.add(self.hero['hero' + str(i)])

    def set_enemy_image(self):
        self.enemy = self.__dict__
        self.enemy_group = pygame.sprite.Group()
        self.enemy_image = self.GRAPHICS['fighter-green']
        for i in range(self.enemy_num):
            self.enemy['enemy' + str(i)] = player.Enemy(image=self.enemy_image)
            self.enemy_group.add(self.enemy['enemy' + str(i)])

    def set_obstacle_image(self):
        self.obstacle = self.__dict__
        self.obstacle_group = pygame.sprite.Group()
        self.obstacle_image = self.GRAPHICS['hole']
        for i, (x, y) in enumerate(self.obstacle_positions):
            obs = player.Obstacle(image=self.obstacle_image)
            obs.init_x = x
            obs.init_y = y
            obs.rect = obs.image.get_rect(center=(x, y))
            self.obstacle[f'obstacle{i}'] = obs
            self.obstacle_group.add(obs)

    def set_goal_image(self):
        self.goal = self.__dict__
        self.goal_group = pygame.sprite.Group()
        self.goal_image = self.GRAPHICS['goal']
        for i, (x, y) in enumerate(self.goal_positions):
            goal = player.Goal(image=self.goal_image)
            goal.init_x = x
            goal.init_y = y
            goal.rect = goal.image.get_rect(center=(x, y))
            self.goal[f'goal{i}'] = goal
            self.goal_group.add(goal)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reached_goal_num = 0
        self.done = False
        self.done_reason = 'other'
        if self.Render:
            self.start()
        else:
            self.set_hero()
            self.set_enemy()
            self.generate_random_obstacles()
            self.set_obstacle()
            self.goal_positions = self.generate_random_goals()
            self.set_goal()
        self.team_counter = 0
        self.hero_state = np.zeros((self.hero_num + self.enemy_num, 25))  # 状态维度自动适配

        hero0 = self.hero['hero0']
        enemy0 = self.enemy['enemy0']

        # 构造状态：自身4维 + 1个障碍物(2) + 5个终点(10) + 完成度(1) = 18 → 扩展为25兼容
        hero_state = [
            hero0.init_x / 1000, hero0.init_y / 1000, hero0.speed / 30, hero0.theta * 57.3 / 360,
            self.obstacle_positions[0][0]/1000, self.obstacle_positions[0][1]/1000,
        ]
        for (gx, gy) in self.goal_positions:
            hero_state.append(gx/1000)
            hero_state.append(gy/1000)
        hero_state.append(self.reached_goal_num / self.goal_num)
        hero_state += [0]*(25 - len(hero_state))

        enemy_state = [
            enemy0.init_x / 1000, enemy0.init_y / 1000, enemy0.speed / 30, enemy0.theta * 57.3 / 360,
            self.obstacle_positions[0][0]/1000, self.obstacle_positions[0][1]/1000,
        ]
        for (gx, gy) in self.goal_positions:
            enemy_state.append(gx/1000)
            enemy_state.append(gy/1000)
        enemy_state.append(self.reached_goal_num / self.goal_num)
        enemy_state += [0]*(25 - len(enemy_state))

        self.hero_state[0] = np.array(hero_state)
        self.hero_state[1] = np.array(enemy_state)
        return self.hero_state, {}

    def get_nearest_obstacle_distance(self, x, y):
        min_dist = float('inf')
        collide = 0
        for (ox, oy) in self.obstacle_positions:
            d = math.hypot(x - ox, y - oy)
            if d < min_dist:
                min_dist = d
            if d < 20:
                collide = 1
        return min_dist, collide

    def check_goal_reach(self, x, y):
        for i in range(self.goal_num):
            if self.goal_exists[i]:
                gx, gy = self.goal_positions[i]
                d = math.hypot(x - gx, y - gy)
                if d < 40:
                    self.goal_exists[i] = False
                    self.reached_goal_num += 1
                    return True
        return False

    def check_edge_penalty(self, x, y):
        if x <= self.MIN_X or x >= self.MAX_X:
            return self.EDGE_PENALTY
        if y <= self.MIN_Y or y >= self.MAX_Y:
            return self.EDGE_PENALTY
        return 0

    def step(self, action):
        r = np.zeros(self.hero_num + self.enemy_num)
        edge_r = np.zeros(self.hero_num)
        edge_r_f = np.zeros(self.enemy_num)
        obstacle_r = np.zeros(self.hero_num)
        obstacle_r1 = np.zeros(self.enemy_num)
        goal_r = np.zeros(self.hero_num)
        goal_r_f = np.zeros(self.enemy_num)
        follow_r = np.zeros(self.enemy_num)
        follow_r0 = 0

        self.done = False
        self.done_reason = 'other'
        h = self.hero['hero0']
        f = self.enemy['enemy0']
        dis_leader_follower = math.hypot(h.posx - f.posx, h.posy - f.posy)

        # ---------------- 领航者 ----------------
        d, _ = self.get_nearest_obstacle_distance(h.posx, h.posy)
        edge_r[0] = self.check_edge_penalty(h.posx, h.posy)
        min_g = float('inf')
        for j in range(self.goal_num):
            if self.goal_exists[j]:
                min_g = min(min_g, math.hypot(h.posx - self.goal_positions[j][0], h.posy - self.goal_positions[j][1]))

        if 0 < dis_leader_follower < 50:
            follow_r0 = 0
            self.team_counter += 1
        else:
            follow_r0 = -0.001 * dis_leader_follower

        if not h.dead:
            if d < 20:
                obstacle_r[0] = -500
                h.die()
                self.done = True
                self.done_reason = 'hero_collision'
            elif d < 40:
                obstacle_r[0] = -2
            else:
                self.check_goal_reach(h.posx, h.posy)
                if self.reached_goal_num == self.goal_num:
                    goal_r[0] = 1000.0
                    h.win = True
                    h.die()
                    self.done = True
                    self.done_reason = 'all_goals_reached'
                else:
                    goal_r[0] = -0.001 * min_g

        r[0] = edge_r[0] + obstacle_r[0] + goal_r[0] + follow_r0
        h.update(action[0], self.Render)

        # ---------------- 跟随者（速度强制同步） ----------------
        f.speed = h.speed
        d, _ = self.get_nearest_obstacle_distance(f.posx, f.posy)
        edge_r_f[0] = self.check_edge_penalty(f.posx, f.posy)
        min_g = float('inf')
        for j in range(self.goal_num):
            if self.goal_exists[j]:
                min_g = min(min_g, math.hypot(f.posx - self.goal_positions[j][0], f.posy - self.goal_positions[j][1]))

        if not f.dead:
            if d < 20:
                obstacle_r1[0] = -500
                f.die()
                self.done = True
                self.done_reason = 'follower_collision'
                r[1] = obstacle_r1[0]
            elif d < 40:
                obstacle_r1[0] = -2
            else:
                self.check_goal_reach(f.posx, f.posy)
                if self.reached_goal_num == self.goal_num:
                    goal_r_f[0] = 1000.0
                    f.win = True
                    f.die()
                    self.done = True
                    self.done_reason = 'all_goals_reached'
                else:
                    goal_r_f[0] = -0.001 * min_g

                if 0 < dis_leader_follower < 50:
                    follow_r[0] = 0
                else:
                    follow_r[0] = -0.001 * dis_leader_follower

                r[1] = edge_r_f[0] + obstacle_r1[0] + follow_r[0] + goal_r_f[0]
        else:
            r[1] = -100.0

        f.update(action[1], self.Render)

        # ---------------- 状态更新 ----------------
        hero_state = [
            h.posx/1000, h.posy/1000, h.speed/30, h.theta*57.3/360,
            self.obstacle_positions[0][0]/1000, self.obstacle_positions[0][1]/1000,
        ]
        for gx, gy in self.goal_positions:
            hero_state.append(gx/1000)
            hero_state.append(gy/1000)
        hero_state.append(self.reached_goal_num / self.goal_num)
        hero_state += [0]*(25 - len(hero_state))
        self.hero_state[0] = np.array(hero_state)

        enemy_state = [
            f.posx/1000, f.posy/1000, f.speed/30, f.theta*57.3/360,
            self.obstacle_positions[0][0]/1000, self.obstacle_positions[0][1]/1000,
        ]
        for gx, gy in self.goal_positions:
            enemy_state.append(gx/1000)
            enemy_state.append(gy/1000)
        enemy_state.append(self.reached_goal_num / self.goal_num)
        enemy_state += [0]*(25 - len(enemy_state))
        self.hero_state[1] = np.array(enemy_state)

        info = {
            'win': h.win or f.win,
            'team_counter': self.team_counter,
            'done_reason': self.done_reason,
            'reached_goal_num': self.reached_goal_num
        }

        if self.done:
            reason = {'all_goals_reached':'所有终点已到达','hero_collision':'领导者撞障碍物','follower_collision':'跟随者撞障碍物'}.get(self.done_reason,'结束')
            print(f"✅ 回合结束 | 已到达终点：{self.reached_goal_num}/{self.goal_num} | 原因：{reason}")

        return self.hero_state, r, self.done, False, info

    def render(self):
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                exit()
        self.SCREEN.blit(self.battle_background, self.view)
        self.obstacle_group.draw(self.SCREEN)
        for i in range(self.goal_num):
            if self.goal_exists[i]:
                g = self.goal[f'goal{i}']
                self.SCREEN.blit(g.image, g.rect)
        self.hero_group.draw(self.SCREEN)
        self.enemy_group.draw(self.SCREEN)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.Render:
            pygame.quit()