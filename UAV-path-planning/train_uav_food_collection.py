# 第一步：环境配置与模块导入
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Win11兼容设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['SUPPRESS_MA_PROMPT'] = '1'

# 添加multiagent路径（替换成你的实际路径）
sys.path.append("C:\\Users\\22895\\Desktop\\uav_project\\multiagent-particle-envs")
os.chdir('C:\\Users\\22895\\Desktop\\uav_project\\UAV-path-planning')

# 导入多智能体环境模块
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from multiagent.multi_discrete import MultiDiscrete


# ========== 核心工具函数（新增碰撞检测） ==========
def is_collision(agent1, agent2):
    """通用碰撞检测函数（复制自food_collection.py）"""
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False


# ========== CTDE核心网络定义（单智能体适配版） ==========
# Actor网络：局部观测 -> 局部动作
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample(self, state):
        logits = self.forward(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action, log_prob, entropy


# Critic网络：全局状态 + 动作 -> 全局价值（单智能体版）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 输入：状态 + 动作（单智能体无需拼接多智能体信息）
        input_dim = state_dim + action_dim

        # 双Critic结构（避免过估计）
        self.fc1_1 = nn.Linear(input_dim, 512)
        self.fc1_2 = nn.Linear(512, 256)
        self.fc1_3 = nn.Linear(256, 1)

        self.fc2_1 = nn.Linear(input_dim, 512)
        self.fc2_2 = nn.Linear(512, 256)
        self.fc2_3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # 单智能体：直接拼接状态和动作（dim=1是特征维度，不会越界）
        x = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.fc1_3(x1)

        x2 = F.relu(self.fc2_1(x))
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.fc2_3(x2)
        return q1, q2

    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc1_2(x))
        q1 = self.fc1_3(x)
        return q1


# ========== CTDE训练器（单智能体适配版） ==========
class EPCTrainer:
    def __init__(self, state_dim, action_dim, max_action, lr=1e-4, gamma=0.99, entropy_coeff=0.0001):
        self.gamma = gamma

        # 1. Actor网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # 2. Critic网络（单智能体版）
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 3. 优化器（提升版学习率）
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.entropy_coeff = entropy_coeff
        self.tau = 0.006  # 目标网络更新系数（微调提升版）

    def update(self, batch):
        # 1. 提取批量数据（单智能体维度：[batch, state_dim]）
        states = torch.FloatTensor(batch['states'])  # [batch, state_dim]
        next_states = torch.FloatTensor(batch['next_states'])  # [batch, state_dim]
        actions = torch.FloatTensor(batch['actions'])  # [batch, action_dim]
        rewards = torch.FloatTensor(batch['rewards'])  # [batch, 1]
        dones = torch.FloatTensor(batch['dones'])  # [batch, 1]

        # 2. 计算目标Q值（单智能体版）
        with torch.no_grad():
            # 目标Actor生成下一个动作
            next_actions, _, _ = self.actor_target.sample(next_states)
            # 转one-hot编码（匹配输入维度）
            next_actions_oh = F.one_hot(next_actions.long(), num_classes=actions.shape[1]).float()

            # 双Critic取最小值（避免过估计）
            target_q1, target_q2 = self.critic_target(next_states, next_actions_oh)
            target_q = torch.min(target_q1, target_q2)

            # 奖励计算（单智能体无需分配）
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 3. 更新Critic
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.15)  # 梯度裁剪
        self.critic_optimizer.step()

        # 4. 更新Actor（单智能体版）
        actions_pred, log_probs, entropies = self.actor.sample(states)
        actions_pred_oh = F.one_hot(actions_pred.long(), num_classes=actions.shape[1]).float()

        q1, q2 = self.critic(states, actions_pred_oh)
        q = torch.min(q1, q2)

        # Actor损失：最大化Q值 + 熵正则（平衡探索利用）
        actor_loss = (self.entropy_coeff * entropies - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.15)  # 梯度裁剪
        self.actor_optimizer.step()

        # 5. 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }


# ========== 奖励函数优化（提升版） ==========
def enhanced_reward(agent, world):
    reward = 0.0
    food_collected = 0

    # 1. 碰撞惩罚（弱化，避免开局奖励暴跌）
    if agent.collide:
        for landmark in world.landmarks:
            if 'obstacle' in landmark.name and is_collision(agent, landmark):
                reward -= 0.02  # 轻微惩罚

    # 2. 收集食物奖励 + 累积奖励（强化版）
    for landmark in world.landmarks:
        if 'food' in landmark.name and is_collision(agent, landmark):
            food_collected += 1
            reward += 4.0  # 基础收集奖励
            reward += food_collected * 1.5  # 累积奖励上调（1.2→1.5）
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

    # 3. 向最近食物移动奖励（强化引导）
    active_foods = [l for l in world.landmarks if 'food' in l.name]
    if active_foods:
        closest_food = min(active_foods, key=lambda f: np.linalg.norm(agent.state.p_pos - f.state.p_pos))
        dist = np.linalg.norm(agent.state.p_pos - closest_food.state.p_pos)
        reward += max(0, 0.15 - 0.03 * dist)  # 引导强度上调（0.12→0.15，0.028→0.03）

    # 4. 靠近障碍物惩罚（轻微）
    for landmark in world.landmarks:
        if 'obstacle' in landmark.name:
            obs_dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            if obs_dist < 1.0:
                reward -= (1.0 - obs_dist) * 0.005

    # 5. 保持移动奖励（鼓励探索）
    agent_speed = np.linalg.norm(agent.state.p_vel)
    if agent_speed > 0.01:
        reward += 0.001

    return reward


# ========== 工具函数 ==========
def is_valid(value):
    """检查数值是否有效（非无穷/非NaN）"""
    if isinstance(value, np.ndarray):
        return not (np.isinf(value).any() or np.isnan(value).any())
    elif isinstance(value, torch.Tensor):
        return not (torch.isinf(value).any() or torch.isnan(value).any())
    else:
        return not (np.isinf(value) or np.isnan(value))


def one_hot_action(action, num_actions):
    """将离散动作转为one-hot编码"""
    oh = np.zeros(num_actions, dtype=np.float32)
    oh[int(action)] = 1.0
    return oh


def calculate_smooth_reward(rewards, window=10):
    """计算平滑奖励（窗口平均）"""
    if len(rewards) < window:
        return np.mean(rewards) if rewards else 0.0
    return np.mean(rewards[-window:])


# ========== 主训练流程（单智能体专用版） ==========
if __name__ == "__main__":
    # 1. 配置参数（提升版）
    num_episodes = 2000  # 训练回合数
    ep_length = 300  # 每回合步数
    batch_size = 64  # 批量大小
    gamma = 0.99  # 折扣因子
    lr = 1e-5  # 学习率上调（8e-6→1e-5）
    ent_coef = 0.0001  # 熵系数上调（0.00005→0.0001）
    save_dir = "./uav_epc_results"  # 结果保存目录

    # 2. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 测试路径可写性
    test_save_dir = os.path.join(os.getcwd(), "train_results")
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, "test.txt"), "w") as f:
        f.write("测试文件：路径可写")
    print(f"测试文件已生成：{os.path.join(test_save_dir, 'test.txt')}")

    # 3. 加载环境
    scenario = scenarios.load("food_collection.py").Scenario()
    scenario.reward = enhanced_reward  # 替换为优化后的奖励函数
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # 4. 初始化维度（单智能体版）
    num_agents = len(env.agents)  # 智能体数量（这里是1）
    state_dim = env.observation_space[0].shape[0]

    # 动作维度适配
    if hasattr(env.action_space[0], 'n'):
        num_actions = env.action_space[0].n
        max_action = num_actions - 1
    elif hasattr(env.action_space[0], 'high'):
        num_actions = env.action_space[0].shape[0]
        max_action = env.action_space[0].high[0]
    elif isinstance(env.action_space[0], MultiDiscrete):
        num_actions = env.action_space[0].num_discrete_space
        max_action = env.action_space[0].high.max()
    else:
        num_actions = 1
        max_action = 1.0

    # 5. 初始化CTDE训练器（提升版）
    trainer = EPCTrainer(
        state_dim=state_dim,
        action_dim=num_actions,
        max_action=max_action,
        lr=lr,
        gamma=gamma,
        entropy_coeff=ent_coef
    )

    # 6. 经验回放池（扩大容量+延迟清理）
    replay_buffer = deque(maxlen=40000)

    # 7. 日志记录
    train_logs = {
        'episodes': [],
        'total_rewards': [],
        'actor_losses': [],
        'critic_losses': [],
        'smooth_rewards': []
    }

    # 8. 开始训练
    print("===== 无人机Food Collection训练开始（CTDE提升版，2000回合） =====")
    print(f"配置：lr={lr}, 熵系数={ent_coef}, 梯度裁剪max_norm=0.15")
    print(f"智能体数量：{num_agents}, 状态维度：{state_dim}, 动作维度：{num_actions}")

    for episode in range(num_episodes):
        # 重置环境
        obs = env.reset()
        total_reward = 0.0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        step_count = 0

        for step in range(ep_length):
            # 生成动作（单智能体版）
            ob = obs[0]  # 只取第一个智能体的观测
            action_tensor, _, _ = trainer.actor.sample(torch.FloatTensor(ob.astype(np.float32)))
            action = int(torch.clamp(action_tensor.float().round(), 0, max_action).item())
            actions = [action]  # 适配环境的动作格式（列表）
            action_oh = one_hot_action(action, num_actions)

            # 执行动作
            next_obs, rewards, dones, _ = env.step(actions)
            total_reward += rewards[0]  # 只取第一个智能体的奖励
            step_count += 1

            # 存储经验（单智能体版：维度简化）
            replay_buffer.append({
                'states': obs[0],  # [state_dim]
                'actions': action_oh,  # [action_dim]
                'next_states': next_obs[0],  # [state_dim]
                'rewards': rewards[0],  # 标量
                'dones': dones[0]  # 标量
            })

            # 训练更新
            if len(replay_buffer) >= batch_size:
                try:
                    # 采样批量数据（单智能体维度适配）
                    batch_samples = random.sample(replay_buffer, batch_size)
                    batch = {
                        'states': np.stack([s['states'] for s in batch_samples]),  # [batch, state_dim]
                        'actions': np.stack([s['actions'] for s in batch_samples]),  # [batch, action_dim]
                        'next_states': np.stack([s['next_states'] for s in batch_samples]),  # [batch, state_dim]
                        'rewards': np.array([[s['rewards']] for s in batch_samples], dtype=np.float32),  # [batch, 1]
                        'dones': np.array([[s['dones']] for s in batch_samples], dtype=np.float32)  # [batch, 1]
                    }

                    # 检查数据有效性
                    if not is_valid(batch['rewards']):
                        print(f"⚠️  第{episode}回合第{step}步：奖励异常，跳过")
                        continue

                    # 更新网络
                    loss = trainer.update(batch)
                    if is_valid(loss['actor_loss']) and is_valid(loss['critic_loss']):
                        actor_loss_sum += loss['actor_loss']
                        critic_loss_sum += loss['critic_loss']

                except Exception as e:
                    print(f"⚠️  第{episode}回合第{step}步：更新失败 - {str(e)}")
                    continue

            # 更新观测
            obs = next_obs
            if all(dones):
                break

        # 计算平均损失和平滑奖励
        avg_actor_loss = actor_loss_sum / step_count if step_count > 0 else 0
        avg_critic_loss = critic_loss_sum / step_count if step_count > 0 else 0
        smooth_reward = calculate_smooth_reward(train_logs['total_rewards'] + [total_reward])

        # 记录日志
        train_logs['episodes'].append(episode)
        train_logs['total_rewards'].append(total_reward)
        train_logs['actor_losses'].append(avg_actor_loss)
        train_logs['critic_losses'].append(avg_critic_loss)
        train_logs['smooth_rewards'].append(smooth_reward)

        # 打印进度+保存结果
        if episode % 50 == 0:
            # 早停检查（防止奖励暴跌）
            if len(train_logs['smooth_rewards']) >= 1000:
                recent_smooth = train_logs['smooth_rewards'][-1000:]
                recent_avg = np.mean(recent_smooth)
                best_avg = np.mean(train_logs['smooth_rewards'][-2000:-1000]) if len(
                    train_logs['smooth_rewards']) >= 2000 else 0
                if best_avg > 0 and recent_avg < best_avg * 0.85:
                    print("⚠️  平滑奖励连续1000回合下降超15%，触发早停！")
                    torch.save({
                        'actor': trainer.actor.state_dict(),
                        'critic': trainer.critic.state_dict(),
                        'episode': episode,
                        'train_logs': train_logs
                    }, os.path.join(save_dir, 'uav_epc_best_earlystop.pth'))
                    np.save(os.path.join(save_dir, 'train_logs_earlystop.npy'), train_logs)
                    env.close()
                    sys.exit(0)

            # 打印训练信息
            print(
                f"回合 {episode:4d} | 总奖励: {total_reward:5.1f} | 平滑奖励: {smooth_reward:5.1f} | Actor损失: {avg_actor_loss:.4f} | Critic损失: {avg_critic_loss:.4f}")

            # 保存日志和模型
            np.save(os.path.join(save_dir, "train_logs_final_ctde_boost.npy"), train_logs)
            torch.save({
                'actor': trainer.actor.state_dict(),
                'critic': trainer.critic.state_dict(),
                'episode': episode,
                'train_config': {
                    'lr': lr, 'ent_coef': ent_coef, 'gamma': gamma
                }
            }, os.path.join(save_dir, f'uav_epc_ep{episode}_ctde_boost.pth'))

    # 9. 训练结束：保存最终结果
    env.close()
    best_episode = np.argmax(train_logs['smooth_rewards'])
    torch.save({
        'actor': trainer.actor.state_dict(),
        'critic': trainer.critic.state_dict(),
        'best_episode': best_episode,
        'best_smooth_reward': train_logs['smooth_rewards'][best_episode],
        'train_logs': train_logs,
        'train_config': {
            'lr': lr, 'ent_coef': ent_coef, 'gamma': gamma
        }
    }, os.path.join(save_dir, 'uav_epc_best_final_ctde_boost.pth'))

    # 打印最终结果
    print(f"\n===== 2000回合CTDE提升版训练完成！ =====")
    print(f"📊 训练结果：")
    print(f"   - 最优平滑奖励回合：{best_episode}，值：{train_logs['smooth_rewards'][best_episode]:.1f}")
    print(f"   - 最终100回合平均奖励：{np.mean(train_logs['total_rewards'][-100:]):.1f}")
    print(f"   - 奖励峰值：{np.max(train_logs['total_rewards']):.1f}")