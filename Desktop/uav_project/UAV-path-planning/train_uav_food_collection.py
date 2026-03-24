# 第一步：环境配置与模块导入（优先解决中文乱码）
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import warnings

# ========== 核心修复1：彻底解决中文乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'  # 白色背景，避免中文模糊

# 关闭警告+优化运行效率
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['SUPPRESS_MA_PROMPT'] = '1'
plt.switch_backend('TkAgg')  # 适配Windows系统

# ========== 路径配置 ==========
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_script_dir)
multiagent_path = os.path.join(parent_dir, "multiagent-particle-envs")
multiagent_path = os.path.abspath(multiagent_path)
sys.path.append(multiagent_path)
os.chdir(current_script_dir)

# 打印路径信息
print(f"当前工作目录：{current_script_dir}")
print(f"Multiagent路径是否存在：{os.path.exists(multiagent_path)}")

# 导入环境模块
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv


# ========== 无人机动力学（优化：增强转向灵活性，避免死胡同） ==========
class UAVDynamics:
    def __init__(self):
        self.max_speed = 0.18  # 适度提高速度，增强机动性
        self.max_acc = 0.07  # 提高加速度，让转向更灵敏
        self.dt = 0.1  # 时间步长（s）
        self.boundary_threshold = BOUNDARY_THRESHOLD  # 硬边界
        self.soft_boundary_ratio = 0.8  # 软边界=硬边界×0.8
        self.obstacle_avoid_acc = 0.03  # 额外避障加速度（主动远离障碍物）

    def update_state(self, current_pos, current_vel, action, nearest_obs_rel_pos):
        # 动作映射（上下左右）
        action_map = {
            0: np.array([0.0, self.max_acc]),  # 上（y+）
            1: np.array([0.0, -self.max_acc]),  # 下（y-）
            2: np.array([-self.max_acc, 0.0]),  # 左（x-）
            3: np.array([self.max_acc, 0.0])  # 右（x+）
        }
        acc = action_map[action]

        # ========== 核心优化1：主动避障加速度（根据障碍物相对位置调整） ==========
        if nearest_obs_rel_pos is not None and np.linalg.norm(nearest_obs_rel_pos) < OBSTACLE_SAFE_DIST:
            # 计算远离障碍物的方向（与相对位置反向）
            avoid_dir = -nearest_obs_rel_pos / (np.linalg.norm(nearest_obs_rel_pos) + 1e-6)
            # 增加额外避障加速度，主动远离
            acc += avoid_dir * self.obstacle_avoid_acc

        # 1. 速度预更新
        new_vel = current_vel + acc * self.dt

        # 2. 软边界减速
        soft_bound = self.boundary_threshold * self.soft_boundary_ratio
        # x轴（左右边界）
        if current_pos[0] >= soft_bound:
            decay_ratio = (self.boundary_threshold - current_pos[0]) / (self.boundary_threshold - soft_bound)
            new_vel[0] *= max(0.2, decay_ratio)
            if current_pos[0] >= self.boundary_threshold:
                new_vel[0] = -new_vel[0] * 0.6
        elif current_pos[0] <= -soft_bound:
            decay_ratio = (current_pos[0] + self.boundary_threshold) / (self.boundary_threshold - soft_bound)
            new_vel[0] *= max(0.2, decay_ratio)
            if current_pos[0] <= -self.boundary_threshold:
                new_vel[0] = -new_vel[0] * 0.6

        # y轴（上下边界）
        if current_pos[1] >= soft_bound:
            decay_ratio = (self.boundary_threshold - current_pos[1]) / (self.boundary_threshold - soft_bound)
            new_vel[1] *= max(0.2, decay_ratio)
            if current_pos[1] >= self.boundary_threshold:
                new_vel[1] = -new_vel[1] * 0.6
        elif current_pos[1] <= -soft_bound:
            decay_ratio = (current_pos[1] + self.boundary_threshold) / (self.boundary_threshold - soft_bound)
            new_vel[1] *= max(0.2, decay_ratio)
            if current_pos[1] <= -self.boundary_threshold:
                new_vel[1] = -new_vel[1] * 0.6

        # 3. 速度上限约束
        new_vel = np.clip(new_vel, -self.max_speed, self.max_speed)

        # 4. 位置更新（双重保险）
        new_pos = current_pos + new_vel * self.dt + 0.5 * acc * (self.dt ** 2)
        new_pos = np.clip(new_pos, -self.boundary_threshold, self.boundary_threshold)

        return new_pos, new_vel


# ========== 全局配置（优化避障参数） ==========
# 环境参数
BOUNDARY_THRESHOLD = 1.15  # 硬边界
OBSTACLE_RADIUS = 0.08
OBSTACLE_SAFE_DIST = 0.25  # 扩大安全距离，提前触发避障
COLLISION_THRESHOLD = 0.03  # 碰撞判定半径
SOFT_BOUNDARY_DIST = 0.2  # 软边界距离

# 奖励参数（核心优化2：梯度化避障惩罚，强化主动避障）
SURVIVAL_REWARD = 1.0
FOOD_COLLECT_REWARD = 100.0  # 提高食物奖励，引导目标导向
COLLISION_OBSTACLE_PENALTY = -200.0  # 加重碰撞惩罚
SPEED_PENALTY = -2.0
ACC_EXCEED_PENALTY = -5.0
BONUS_SAFE_MOVE = 1.0  # 提高安全移动奖励
BONUS_OBSTACLE_AVOID = 0.8  # 新增：成功远离障碍物的额外奖励
BOUNDARY_PENALTY_MAX = -20.0
BOUNDARY_PENALTY_MIN = -2.0

# SAC算法参数（优化学习率，让策略更快收敛）
EP_MAX = 500
EP_LEN = 1000
GAMMA = 0.95  # 提高折扣因子，重视长期奖励
Q_LR = 4e-4
POLICY_LR = 1.2e-3  # 适度提高策略学习率
ALPHA_LR = 3e-4
BATCH_SIZE = 128
TAU = 1e-2
MEMORY_CAPACITY = 30000  # 扩大经验池，存储更多避障经验
STATE_DIM = 14  # 扩展状态：13维+障碍物距离（1维）
ACTION_DIM = 4
MAX_ACTION = 3
MIN_ACTION = 0
SWITCH = 0  # 0=训练，1=测试

# 初始化动力学
dynamics = UAVDynamics()


# ========== 1. SAC核心网络（适配14维观测） ==========
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.1)  # 新增dropout，避免过拟合
        self.out = nn.Linear(256, action_dim)
        # 权重初始化
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.out.weight, mean=0, std=0.1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

    def sample_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def deterministic_action(self, state):
        logits = self.forward(state)
        action = torch.argmax(logits, dim=-1)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1网络
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_dropout = nn.Dropout(0.1)
        self.q1_out = nn.Linear(256, 1)
        # Q2网络
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_dropout = nn.Dropout(0.1)
        self.q2_out = nn.Linear(256, 1)
        # 权重初始化
        nn.init.normal_(self.q1_fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.q1_fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.q1_out.weight, mean=0, std=0.1)
        nn.init.normal_(self.q2_fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.q2_fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.q2_out.weight, mean=0, std=0.1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        # Q1
        q1 = F.relu(self.q1_fc1(sa))
        q1 = self.q1_dropout(q1)
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        # Q2
        q2 = F.relu(self.q2_fc1(sa))
        q2 = self.q2_dropout(q2)
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2


# ========== 2. 经验池（不变） ==========
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        action_onehot = np.eye(self.action_dim)[action]
        self.buffer.append({
            'state': state.astype(np.float32),
            'action': action_onehot.astype(np.float32),
            'reward': np.array([reward], dtype=np.float32),
            'next_state': next_state.astype(np.float32),
            'done': np.array([float(done)], dtype=np.float32)
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.stack([b['state'] for b in batch]))
        actions = torch.FloatTensor(np.stack([b['action'] for b in batch]))
        rewards = torch.FloatTensor(np.stack([b['reward'] for b in batch]))
        next_states = torch.FloatTensor(np.stack([b['next_state'] for b in batch]))
        dones = torch.FloatTensor(np.stack([b['done'] for b in batch]))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ========== 3. SAC算法核心（不变） ==========
class SAC:
    def __init__(self, state_dim, action_dim):
        # 策略网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=POLICY_LR)
        # Q网络
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=Q_LR)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # 熵系数α
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.alpha = self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1))
        if deterministic:
            action = self.actor.deterministic_action(state)
        else:
            action, _, _ = self.actor.sample_action(state)
        return int(action.item())

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return {'critic_loss': 0, 'actor_loss': 0, 'alpha': self.alpha.item()}

        # 采样经验
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 更新Q网络
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample_action(next_states)
            next_actions_onehot = F.one_hot(next_actions, num_classes=ACTION_DIM).float()
            target_q1, target_q2 = self.critic_target(next_states, next_actions_onehot)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs.unsqueeze(1)
            target_q = rewards + (1 - dones) * GAMMA * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # 更新策略网络
        for param in self.critic.parameters():
            param.requires_grad = False
        pred_actions, pred_log_probs, pred_entropies = self.actor.sample_action(states)
        pred_actions_onehot = F.one_hot(pred_actions, num_classes=ACTION_DIM).float()
        q1, q2 = self.critic(states, pred_actions_onehot)
        q = torch.min(q1, q2)
        actor_loss = - (q + self.alpha * pred_log_probs.unsqueeze(1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # 解冻Q网络
        for param in self.critic.parameters():
            param.requires_grad = True

        # 更新α
        alpha_loss = -(self.log_alpha.exp() * (pred_log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item()
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.alpha = checkpoint['alpha']
        self.log_alpha = checkpoint['log_alpha']


# ========== 4. 工具函数（优化避障逻辑+中文显示） ==========
def print_uav_env_info():
    """告知UAV边界规则"""
    print("\n===== UAV 环境规则 =====")
    print(f"1. 边界：硬边界[{BOUNDARY_THRESHOLD}], 软边界[{SOFT_BOUNDARY_DIST}]，无法飞出")
    print(f"2. 奖励：收集食物+{FOOD_COLLECT_REWARD}，安全移动+{BONUS_SAFE_MOVE}，避障成功+{BONUS_OBSTACLE_AVOID}")
    print(f"3. 惩罚：碰撞-200，靠近边界-2~-20，靠近障碍物-5~-50，速度过慢-2")
    print("===========================\n")


def is_collision(agent_pos, agent_size, target_pos, target_size):
    delta_pos = agent_pos - target_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    return dist < (agent_size + target_size - COLLISION_THRESHOLD)


def calculate_boundary_reward(agent_pos):
    """梯度化边界惩罚"""
    dist_to_right = BOUNDARY_THRESHOLD - agent_pos[0]
    dist_to_left = agent_pos[0] + BOUNDARY_THRESHOLD
    dist_to_top = BOUNDARY_THRESHOLD - agent_pos[1]
    dist_to_bottom = agent_pos[1] + BOUNDARY_THRESHOLD
    min_dist = min(dist_to_right, dist_to_left, dist_to_top, dist_to_bottom)

    if min_dist >= SOFT_BOUNDARY_DIST:
        return 0.0
    else:
        penalty_ratio = (SOFT_BOUNDARY_DIST - min_dist) / SOFT_BOUNDARY_DIST
        return BOUNDARY_PENALTY_MIN - penalty_ratio * (BOUNDARY_PENALTY_MIN - BOUNDARY_PENALTY_MAX)


def calculate_obstacle_reward(agent_pos, agent_size, obstacles):
    """核心优化3：梯度化避障惩罚+避障奖励"""
    min_dist_to_obs = min([np.linalg.norm(agent_pos - obs['pos']) for obs in obstacles]) if obstacles else float('inf')
    reward = 0.0

    # 1. 碰撞惩罚（加重）
    if any([is_collision(agent_pos, agent_size, obs['pos'], obs['size']) for obs in obstacles]):
        reward += COLLISION_OBSTACLE_PENALTY
    # 2. 近距离危险惩罚（梯度化）
    elif min_dist_to_obs < OBSTACLE_SAFE_DIST * 0.5:  # 极近距离（<0.125m）
        reward += -50.0 * (OBSTACLE_SAFE_DIST * 0.5 - min_dist_to_obs) / (OBSTACLE_SAFE_DIST * 0.5)
    elif min_dist_to_obs < OBSTACLE_SAFE_DIST:  # 安全距离内（0.125~0.25m）
        reward += -10.0 * (OBSTACLE_SAFE_DIST - min_dist_to_obs) / OBSTACLE_SAFE_DIST
    # 3. 成功远离障碍物奖励
    elif min_dist_to_obs > OBSTACLE_SAFE_DIST * 1.5:
        reward += BONUS_OBSTACLE_AVOID

    return reward, min_dist_to_obs


def get_boundary_distances(agent_pos):
    """计算到4个边界的距离"""
    return np.array([
        BOUNDARY_THRESHOLD - agent_pos[0],
        agent_pos[0] + BOUNDARY_THRESHOLD,
        BOUNDARY_THRESHOLD - agent_pos[1],
        agent_pos[1] + BOUNDARY_THRESHOLD
    ])


def generate_valid_food_positions(obstacles, num_foods=8):
    """生成边界内不重叠的食物位置"""
    food_positions = []
    max_attempts = 500
    attempts = 0
    while len(food_positions) < num_foods and attempts < max_attempts:
        pos = np.array([
            random.uniform(-BOUNDARY_THRESHOLD + SOFT_BOUNDARY_DIST / 2, BOUNDARY_THRESHOLD - SOFT_BOUNDARY_DIST / 2),
            random.uniform(-BOUNDARY_THRESHOLD + SOFT_BOUNDARY_DIST / 2, BOUNDARY_THRESHOLD - SOFT_BOUNDARY_DIST / 2)
        ])
        overlap = False
        for obs in obstacles:
            if np.linalg.norm(pos - obs['pos']) < 0.3:  # 食物与障碍物保持0.3m距离
                overlap = True
                break
        if not overlap:
            food_positions.append(pos)
        attempts += 1
    return food_positions


def augment_observation(agent_state, all_food_pos, all_obs_pos):
    """核心优化4：扩展观测维度，加入障碍物距离（14维）"""
    vel = agent_state['vel']
    # 最近食物相对位置+距离
    if all_food_pos:
        nearest_food = min(all_food_pos, key=lambda p: np.linalg.norm(p - agent_state['pos']))
        food_rel_pos = nearest_food - agent_state['pos']
        food_dist = np.array([np.linalg.norm(food_rel_pos)])  # 新增：食物距离
    else:
        food_rel_pos = np.array([0.0, 0.0])
        food_dist = np.array([0.0])

    # 最近障碍物相对位置+距离
    nearest_obs_rel_pos = None
    if all_obs_pos:
        nearest_obs = min(all_obs_pos, key=lambda p: np.linalg.norm(p - agent_state['pos']))
        nearest_obs_rel_pos = nearest_obs - agent_state['pos']
        obs_dist = np.array([np.linalg.norm(nearest_obs_rel_pos)])  # 新增：障碍物距离
    else:
        nearest_obs_rel_pos = np.array([0.0, 0.0])
        obs_dist = np.array([0.0])

    # 其他特征
    step_norm = min(1.0, len(agent_state.get('step_log', [0])) / EP_LEN)
    boundary_info = np.array([BOUNDARY_THRESHOLD, BOUNDARY_THRESHOLD])
    boundary_distances = get_boundary_distances(agent_state['pos'])

    # 拼接：14维（原13维+障碍物距离1维）
    obs = np.concatenate([vel, food_rel_pos, food_dist, nearest_obs_rel_pos, obs_dist,
                          [step_norm], boundary_info, boundary_distances])
    return obs[:STATE_DIM].astype(np.float32), nearest_obs_rel_pos


def paper_based_reward(agent_state, foods, obstacles, step, collected_foods):
    """优化奖励函数，强化避障引导"""
    reward = SURVIVAL_REWARD + 0.001 * step
    # 食物收集奖励（提高权重）
    new_collected = []
    for i, food in enumerate(foods):
        if i not in collected_foods and not food['collected'] and is_collision(
                agent_state['pos'], agent_state['size'], food['pos'], food['size']
        ):
            reward += FOOD_COLLECT_REWARD
            new_collected.append(i)
            food['collected'] = True
    # 速度奖励
    speed = np.linalg.norm(agent_state['vel'])
    if speed < 0.03:
        reward += SPEED_PENALTY
    elif speed > dynamics.max_speed * 0.6:
        reward += 0.5
    # 避障奖励/惩罚（梯度化）
    obstacle_reward, min_dist_to_obs = calculate_obstacle_reward(agent_state['pos'], agent_state['size'], obstacles)
    reward += obstacle_reward
    # 边界惩罚
    reward += calculate_boundary_reward(agent_state['pos'])
    # 加速度超限惩罚
    if np.any(np.abs(agent_state['acc']) > dynamics.max_acc + dynamics.obstacle_avoid_acc):
        reward += ACC_EXCEED_PENALTY
    # 安全移动奖励
    if reward >= 0:
        reward += BONUS_SAFE_MOVE
    return reward, new_collected


# ========== 5. 可视化工具类（确认中文配置+优化显示） ==========
class UAVVisualizer:
    def __init__(self):
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=80)
        self.fig.suptitle('无人机食物收集（论文适配版）', fontsize=14, fontproperties='SimHei')

        # 轨迹图
        self.ax1.set_xlim(-1.3, 1.3)
        self.ax1.set_ylim(-1.3, 1.3)
        self.ax1.set_xlabel('X坐标 (m)', fontproperties='SimHei')
        self.ax1.set_ylabel('Y坐标 (m)', fontproperties='SimHei')
        self.ax1.set_title('运动轨迹', fontproperties='SimHei')
        self.ax1.grid(True, alpha=0.3)
        # 硬边界
        hard_bound = plt.Rectangle((-BOUNDARY_THRESHOLD, -BOUNDARY_THRESHOLD), 2 * BOUNDARY_THRESHOLD,
                                   2 * BOUNDARY_THRESHOLD,
                                   linewidth=2, edgecolor='orange', facecolor='none', label=f'硬边界')
        self.ax1.add_patch(hard_bound)
        # 软边界
        soft_bound = plt.Rectangle((-BOUNDARY_THRESHOLD + SOFT_BOUNDARY_DIST, -BOUNDARY_THRESHOLD + SOFT_BOUNDARY_DIST),
                                   2 * (BOUNDARY_THRESHOLD - SOFT_BOUNDARY_DIST),
                                   2 * (BOUNDARY_THRESHOLD - SOFT_BOUNDARY_DIST),
                                   linewidth=1.5, edgecolor='yellow', linestyle='--', facecolor='none', label=f'软边界')
        self.ax1.add_patch(soft_bound)
        # 避障安全区（红色虚线）
        avoid_bound = plt.Rectangle(
            (-BOUNDARY_THRESHOLD + OBSTACLE_SAFE_DIST, -BOUNDARY_THRESHOLD + OBSTACLE_SAFE_DIST),
            2 * (BOUNDARY_THRESHOLD - OBSTACLE_SAFE_DIST), 2 * (BOUNDARY_THRESHOLD - OBSTACLE_SAFE_DIST),
            linewidth=1, edgecolor='red', linestyle=':', facecolor='none', label=f'避障安全区')
        self.ax1.add_patch(avoid_bound)
        # 绘图元素
        self.uav_scatter = self.ax1.scatter([0], [0], c='blue', s=80, label='无人机', zorder=5)
        self.path_line, = self.ax1.plot([], [], c='blue', alpha=0.5, linewidth=1)
        self.food_scatters = []
        self.obstacle_scatters = []
        self.ax1.legend(loc='upper right', fontsize=8, prop={'family': 'SimHei'})

        # 奖励图
        self.ax2.set_xlabel('步数', fontproperties='SimHei')
        self.ax2.set_ylabel('奖励值', fontproperties='SimHei')
        self.ax2.set_title('奖励变化', fontproperties='SimHei')
        self.ax2.grid(True, alpha=0.3)
        self.reward_line, = self.ax2.plot([], [], c='orange', alpha=0.5, label='单步奖励')
        self.cumulative_line, = self.ax2.plot([], [], c='green', linewidth=2, label='累计奖励')
        self.ax2.legend(loc='upper right', fontsize=8, prop={'family': 'SimHei'})

        # 障碍物距离图（新增：直观显示避障效果）
        self.ax3.set_xlabel('步数', fontproperties='SimHei')
        self.ax3.set_ylabel('距离 (m)', fontproperties='SimHei')
        self.ax3.set_title('到最近障碍物距离', fontproperties='SimHei')
        self.ax3.grid(True, alpha=0.3)
        self.obs_dist_line, = self.ax3.plot([], [], c='red', linewidth=1.5, label='障碍物距离')
        self.safe_dist_line = self.ax3.axhline(y=OBSTACLE_SAFE_DIST, color='red', linestyle='--', alpha=0.7,
                                               label='安全距离')
        self.ax3.legend(loc='upper right', fontsize=8, prop={'family': 'SimHei'})
        self.obs_dist_log = []

        # 速度图
        self.ax4.set_xlabel('步数', fontproperties='SimHei')
        self.ax4.set_ylabel('速度 (m/s)', fontproperties='SimHei')
        self.ax4.set_title('速度变化', fontproperties='SimHei')
        self.ax4.grid(True, alpha=0.3)
        self.vel_x_line, = self.ax4.plot([], [], c='red', label='x方向')
        self.vel_y_line, = self.ax4.plot([], [], c='blue', label='y方向')
        self.ax4.legend(loc='upper right', fontsize=8, prop={'family': 'SimHei'})
        self.vel_log = []

        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

    def update(self, agent_pos, agent_vel, foods, obstacles, step_reward, step, min_dist_to_obs):
        """每5步更新一次，减少卡顿"""
        if step % 5 != 0 and step != 0:
            return

        # 轨迹更新
        if step == 0:
            self.path_x, self.path_y = [], []
            self.step_rewards, self.cumulative_rewards = [], []
            self.obs_dist_log, self.vel_log = [], []
        self.path_x.append(agent_pos[0])
        self.path_y.append(agent_pos[1])
        self.uav_scatter.set_offsets([agent_pos[0], agent_pos[1]])
        self.path_line.set_data(self.path_x, self.path_y)

        # 食物更新
        while len(self.food_scatters) < len(foods):
            scatter = self.ax1.scatter([], [], c='green', s=40, label='食物' if len(self.food_scatters) == 0 else "",
                                       zorder=3)
            self.food_scatters.append(scatter)
        for i, food in enumerate(foods):
            if i < len(self.food_scatters):
                self.food_scatters[i].set_offsets([food['pos'][0], food['pos'][1]])
                self.food_scatters[i].set_visible(not food['collected'])

        # 障碍物更新
        while len(self.obstacle_scatters) < len(obstacles):
            scatter = self.ax1.scatter([], [], c='red', s=60,
                                       label='障碍物' if len(self.obstacle_scatters) == 0 else "", zorder=4)
            self.obstacle_scatters.append(scatter)
        for i, obs in enumerate(obstacles):
            if i < len(self.obstacle_scatters):
                self.obstacle_scatters[i].set_offsets([obs['pos'][0], obs['pos'][1]])

        # 奖励更新
        self.step_rewards.append(step_reward)
        self.cumulative_rewards.append(sum(self.step_rewards[-50:]))
        self.reward_line.set_data(range(len(self.step_rewards)), self.step_rewards)
        self.cumulative_line.set_data(range(len(self.cumulative_rewards)), self.cumulative_rewards)
        self.ax2.set_xlim(0, len(self.step_rewards))
        if self.cumulative_rewards:
            self.ax2.set_ylim(min(self.cumulative_rewards) - 20, max(self.cumulative_rewards) + 20)

        # 障碍物距离更新
        self.obs_dist_log.append(min_dist_to_obs)
        self.obs_dist_line.set_data(range(len(self.obs_dist_log)), self.obs_dist_log)
        self.ax3.set_xlim(0, len(self.obs_dist_log))
        self.ax3.set_ylim(0, OBSTACLE_SAFE_DIST * 2)

        # 速度更新
        self.vel_log.append(agent_vel)
        vel_arr = np.array(self.vel_log)
        self.vel_x_line.set_data(range(len(vel_arr)), vel_arr[:, 0])
        self.vel_y_line.set_data(range(len(vel_arr)), vel_arr[:, 1])
        self.ax4.set_xlim(0, len(vel_arr))
        self.ax4.set_ylim(-dynamics.max_speed - 0.05, dynamics.max_speed + 0.05)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def reset(self):
        self.path_x, self.path_y = [], []
        self.step_rewards, self.cumulative_rewards = [], []
        self.obs_dist_log, self.vel_log = [], []


# ========== 6. 环境实体初始化 ==========
def init_env_entities(world, num_foods=8):
    """初始化食物/障碍物，避免密集分布导致死胡同"""
    obstacles = []
    for landmark in world.landmarks:
        if 'obstacle' in landmark.name:
            obs_pos = landmark.state.p_pos.copy()
            obs_pos = np.clip(obs_pos, -BOUNDARY_THRESHOLD + OBSTACLE_SAFE_DIST,
                              BOUNDARY_THRESHOLD - OBSTACLE_SAFE_DIST)
            # 确保障碍物之间不密集（避免死胡同）
            if not obstacles or min([np.linalg.norm(obs_pos - o['pos']) for o in obstacles]) > 0.3:
                obstacles.append({'pos': obs_pos, 'size': OBSTACLE_RADIUS})

    # 生成有效食物位置
    food_positions = generate_valid_food_positions(obstacles, num_foods)
    foods = [{'pos': pos, 'size': 0.05, 'collected': False} for pos in food_positions]

    # 记录全量位置
    all_food_pos = [f['pos'] for f in foods]
    all_obs_pos = [o['pos'] for o in obstacles]
    return foods, obstacles, all_food_pos, all_obs_pos


# ========== 7. 训练结果作图 ==========
def plot_training_results(train_logs, save_dir):
    episodes = np.array(train_logs['episodes'], dtype=np.int32)
    total_rewards = np.array(train_logs['total_rewards'], dtype=np.float32)
    actor_losses = np.array(train_logs['actor_losses'], dtype=np.float32)
    critic_losses = np.array(train_logs['critic_losses'], dtype=np.float32)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=100)
    # 奖励曲线
    ax1.plot(episodes, total_rewards, color='#2E8B57', linewidth=1.5, alpha=0.8, label='每回合奖励')
    if len(episodes) >= 10:
        smoothed = np.convolve(total_rewards, np.ones(10) / 10, mode='valid')
        ax1.plot(episodes[9:], smoothed, color='#FF6347', linewidth=2, label='10回合平滑')
    ax1.set_xlabel('训练回合', fontproperties='SimHei')
    ax1.set_ylabel('总奖励', fontproperties='SimHei')
    ax1.set_title('奖励趋势', fontproperties='SimHei')
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop={'family': 'SimHei'})

    # 损失曲线
    ax2.plot(episodes, actor_losses, color='#4169E1', linewidth=1.5, alpha=0.8, label='Actor损失')
    ax2.plot(episodes, critic_losses, color='#DC143C', linewidth=1.5, alpha=0.8, label='Critic损失')
    ax2.set_xlabel('训练回合', fontproperties='SimHei')
    ax2.set_ylabel('损失值', fontproperties='SimHei')
    ax2.set_title('损失趋势', fontproperties='SimHei')
    ax2.grid(True, alpha=0.3)
    ax2.legend(prop={'family': 'SimHei'})

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"训练曲线已保存：{save_path}")
    plt.show()

    # 输出关键指标
    print("\n📊 训练关键指标：")
    print(f"最高奖励：{total_rewards.max():.1f}（回合 {episodes[np.argmax(total_rewards)]}）")
    if len(total_rewards) >= 100:
        print(f"最后100回合平均奖励：{total_rewards[-100:].mean():.1f}")
    print(f"最终Critic损失：{critic_losses[-1]:.4f}")


# ========== 8. 主训练/测试逻辑 ==========
def train_sac():
    print_uav_env_info()
    # 创建保存目录
    save_dir = "./uav_masac_results"
    os.makedirs(save_dir, exist_ok=True)
    # 初始化环境
    scenario = scenarios.load("food_collection.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    agent_size = world.agents[0].size
    # 初始化算法和经验池
    sac_agent = SAC(STATE_DIM, ACTION_DIM)
    replay_buffer = ReplayBuffer(MEMORY_CAPACITY, STATE_DIM, ACTION_DIM)
    visualizer = UAVVisualizer()
    # 训练日志
    train_logs = {'episodes': [], 'total_rewards': [], 'food_collected': [], 'actor_losses': [], 'critic_losses': []}

    print("\n===== 开始训练 =====")
    print(f"配置：{EP_MAX}回合，每回合{EP_LEN}步，批次{BATCH_SIZE}")
    print("=" * 80)
    print(f"{'回合':<6} | {'总奖励':<10} | {'收集食物':<8} | {'α值':<8}")
    print("-" * 80)

    for episode in range(EP_MAX):
        env.reset()
        foods, obstacles, all_food_pos, all_obs_pos = init_env_entities(world)
        collected_foods = set()
        # 初始化无人机状态
        agent_state = {
            'pos': np.array([0.0, 0.0]),
            'vel': np.array([0.0, 0.0]),
            'acc': np.array([0.0, 0.0]),
            'size': agent_size,
            'step_log': []
        }
        total_reward = 0.0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        is_terminated = False
        visualizer.reset()

        for step in range(EP_LEN):
            if is_terminated:
                break
            # 更新步数日志
            agent_state['step_log'].append(step)
            # 生成观测（含障碍物距离）
            obs, nearest_obs_rel_pos = augment_observation(agent_state, all_food_pos, all_obs_pos)
            # 选择动作
            action = sac_agent.select_action(obs)
            # 更新无人机状态（加入主动避障加速度）
            new_pos, new_vel = dynamics.update_state(agent_state['pos'], agent_state['vel'], action,
                                                     nearest_obs_rel_pos)
            agent_state['acc'] = (new_vel - agent_state['vel']) / dynamics.dt
            agent_state['pos'] = new_pos
            agent_state['vel'] = new_vel
            # 更新环境
            world.agents[0].state.p_pos = new_pos
            world.agents[0].state.p_vel = new_vel
            env.step([action])
            # 计算奖励（梯度化避障）
            step_reward, new_collected = paper_based_reward(agent_state, foods, obstacles, step, collected_foods)
            collected_foods.update(new_collected)
            # 计算到最近障碍物距离（用于可视化）
            min_dist_to_obs = min([np.linalg.norm(new_pos - obs['pos']) for obs in obstacles]) if obstacles else float(
                'inf')
            # 终止条件：仅碰撞障碍物
            if any([is_collision(new_pos, agent_size, obs['pos'], obs['size']) for obs in obstacles]):
                is_terminated = True
                print(f"⚠️  回合{episode}第{step}步碰撞障碍物，总奖励：{total_reward:.1f}")
            # 存储经验
            next_obs, _ = augment_observation(agent_state, all_food_pos, all_obs_pos)
            replay_buffer.store(obs, action, step_reward, next_obs, is_terminated)
            # 更新网络
            loss = sac_agent.update(replay_buffer, BATCH_SIZE)
            actor_loss_sum += loss['actor_loss']
            critic_loss_sum += loss['critic_loss']
            # 可视化更新（传入障碍物距离）
            visualizer.update(new_pos, new_vel, foods, obstacles, step_reward, step, min_dist_to_obs)
            # 累计奖励
            total_reward += step_reward

        # 记录日志
        food_count = len([f for f in foods if f['collected']])
        avg_actor_loss = actor_loss_sum / (step + 1) if step > 0 else 0
        avg_critic_loss = critic_loss_sum / (step + 1) if step > 0 else 0
        train_logs['episodes'].append(episode)
        train_logs['total_rewards'].append(total_reward)
        train_logs['food_collected'].append(food_count)
        train_logs['actor_losses'].append(avg_actor_loss)
        train_logs['critic_losses'].append(avg_critic_loss)
        # 打印进度
        print(f"{episode:<6} | {total_reward:<10.1f} | {food_count:<8d} | {sac_agent.alpha.item():<8.4f}")
        # 每50回合保存模型
        if episode % 50 == 0:
            sac_agent.save(os.path.join(save_dir, f'masac_ep{episode}.pth'))

    # 训练完成
    env.close()
    plt.ioff()
    plt.close(visualizer.fig)
    # 保存日志和最优模型
    np.save(os.path.join(save_dir, 'masac_train_logs.npy'), train_logs)
    best_ep = np.argmax(train_logs['total_rewards']) if train_logs['total_rewards'] else 0
    sac_agent.save(os.path.join(save_dir, f'masac_best_ep{best_ep}.pth'))
    # 绘制结果
    if train_logs['episodes']:
        plot_training_results(train_logs, save_dir)
    print(f"\n===== 训练完成 =====")
    if train_logs['total_rewards']:
        print(f"最优回合：{best_ep} | 最优奖励：{train_logs['total_rewards'][best_ep]:.1f}")


def test_sac():
    print_uav_env_info()
    # 初始化环境
    scenario = scenarios.load("food_collection.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    agent_size = world.agents[0].size
    # 加载模型
    sac_agent = SAC(STATE_DIM, ACTION_DIM)
    model_path = "./uav_masac_results/masac_best_ep0.pth"
    if os.path.exists(model_path):
        sac_agent.load(model_path)
    else:
        print(f"警告：模型文件{model_path}不存在，使用随机策略")
    visualizer = UAVVisualizer()

    print("\n===== 开始测试 =====")
    total_rewards = []
    total_food = []
    for episode in range(10):
        env.reset()
        foods, obstacles, all_food_pos, all_obs_pos = init_env_entities(world)
        collected_foods = set()
        agent_state = {
            'pos': np.array([0.0, 0.0]),
            'vel': np.array([0.0, 0.0]),
            'acc': np.array([0.0, 0.0]),
            'size': agent_size,
            'step_log': []
        }
        total_reward = 0.0
        is_terminated = False
        visualizer.reset()

        for step in range(EP_LEN):
            if is_terminated:
                break
            agent_state['step_log'].append(step)
            obs, nearest_obs_rel_pos = augment_observation(agent_state, all_food_pos, all_obs_pos)
            action = sac_agent.select_action(obs, deterministic=True)
            # 更新状态（主动避障）
            new_pos, new_vel = dynamics.update_state(agent_state['pos'], agent_state['vel'], action,
                                                     nearest_obs_rel_pos)
            agent_state['acc'] = (new_vel - agent_state['vel']) / dynamics.dt
            agent_state['pos'] = new_pos
            agent_state['vel'] = new_vel
            world.agents[0].state.p_pos = new_pos
            world.agents[0].state.p_vel = new_vel
            env.step([action])
            # 计算奖励
            step_reward, new_collected = paper_based_reward(agent_state, foods, obstacles, step, collected_foods)
            collected_foods.update(new_collected)
            # 计算障碍物距离
            min_dist_to_obs = min([np.linalg.norm(new_pos - obs['pos']) for obs in obstacles]) if obstacles else float(
                'inf')
            # 终止条件
            if any([is_collision(new_pos, agent_size, obs['pos'], obs['size']) for obs in obstacles]):
                is_terminated = True
                print(f"⚠️  测试回合{episode}第{step}步碰撞障碍物")
            # 可视化
            visualizer.update(new_pos, new_vel, foods, obstacles, step_reward, step, min_dist_to_obs)
            total_reward += step_reward

        food_count = len([f for f in foods if f['collected']])
        total_rewards.append(total_reward)
        total_food.append(food_count)
        print(f"测试回合 {episode} | 奖励：{total_reward:.1f} | 收集食物：{food_count}个")

    env.close()
    plt.ioff()
    plt.show()
    print(f"\n===== 测试结果 =====")
    print(f"平均奖励：{np.mean(total_rewards):.1f} | 平均收集食物：{np.mean(total_food):.1f}个")


# ========== 主函数 ==========
if __name__ == "__main__":
    if SWITCH == 0:
        train_sac()
    else:
        test_sac()