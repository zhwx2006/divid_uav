# ========== 必要导入 ==========
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
import matplotlib.patches as patches
from importlib import import_module

# Win11兼容 + 屏蔽警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings

warnings.filterwarnings("ignore")

# ========== 路径配置 ==========
MPE_PATH = "C:\\Users\\22895\\Desktop\\uav_project\\multiagent-particle-envs"
SCENARIOS_PATH = os.path.join(MPE_PATH, "multiagent", "scenarios")
sys.path.append(MPE_PATH)
sys.path.append(SCENARIOS_PATH)
os.chdir('C:\\Users\\22895\\Desktop\\uav_project\\UAV-path-planning')

# 导入核心模块
from multiagent.environment import MultiAgentEnv
from multiagent.core import World

food_collection = import_module("food_collection")
Scenario = food_collection.Scenario


# ========== 可视化函数 ==========
def init_visualization():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    plt.ion()
    return fig, ax


def visualize_uav(ax, agent, world, episode, step, reward, trajectory):
    ax.clear()
    ax.set_title(f'Episode {episode} | Step {step} | Reward: {reward:.1f}')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)

    # 绘制边界
    min_b, max_b = world.boundary
    ax.axvline(x=min_b, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=max_b, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=min_b, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=max_b, color='r', linestyle='--', alpha=0.5)

    # 绘制障碍物
    for landmark in world.landmarks:
        if 'obstacle' in landmark.name:
            ax.add_patch(plt.Circle(
                (landmark.state.p_pos[0], landmark.state.p_pos[1]),
                landmark.size, color='gray', alpha=0.7, label='Obstacle'
            ))

    # 绘制食物
    for landmark in world.landmarks:
        if 'food' in landmark.name:
            ax.add_patch(plt.Circle(
                (landmark.state.p_pos[0], landmark.state.p_pos[1]),
                landmark.size, color='green', alpha=0.7, label='Food'
            ))

    # 绘制轨迹和无人机
    if len(trajectory) > 1:
        ax.plot([p[0] for p in trajectory], [p[1] for p in trajectory], 'b-', alpha=0.5, label='Trajectory')
    ax.scatter(agent.state.p_pos[0], agent.state.p_pos[1], color='red', s=50, marker='^', label='UAV (Center)')
    ax.legend(loc='upper right')
    plt.draw()
    plt.pause(0.001)


# ========== EPC训练器定义 ==========
class EPCTrainer:
    def __init__(self, state_dim, num_actions, max_action, lr=1e-5, gamma=0.99, ent_coef=0.0001):
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.max_action = max_action

        # 演员网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        # 评论家网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim + num_actions, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)

    def sample(self, state):
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def update(self, batch):
        states = torch.FloatTensor(batch['states'])
        actions = torch.FloatTensor(batch['actions'])
        next_states = torch.FloatTensor(batch['next_states'])
        rewards = torch.FloatTensor(batch['rewards'])
        dones = torch.FloatTensor(batch['dones'])

        # 计算目标Q值
        next_q = self.critic(next_states, self.actor(next_states))
        target_q = rewards + (1 - dones) * self.gamma * next_q.detach()

        # 更新评论家
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 更新演员
        action_logits = self.actor(states)
        dist = torch.distributions.Categorical(logits=action_logits)
        action_sample = dist.sample()
        action_onehot = F.one_hot(action_sample, num_classes=action_logits.shape[-1]).float()
        q_val = self.critic(states, action_onehot)
        actor_loss = - (q_val + self.ent_coef * dist.entropy()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }


# ========== 主训练流程 ==========
if __name__ == "__main__":
    # 配置参数
    num_episodes = 2000
    ep_length = 300
    batch_size = 64
    lr = 1e-5
    gamma = 0.99
    ent_coef = 0.0001
    save_dir = "./uav_epc_results"
    os.makedirs(save_dir, exist_ok=True)

    # 加载环境
    scenario = Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)

    # 维度初始化
    state_dim = env.observation_space[0].shape[0]
    action_space = env.action_space[0]
    num_actions = action_space.n if hasattr(action_space, 'n') else 4
    max_action = num_actions - 1

    # 初始化训练器和回放池
    trainer = EPCTrainer(state_dim, num_actions, max_action, lr, gamma, ent_coef)
    replay_buffer = deque(maxlen=80000)

    # 初始化可视化
    fig, ax = init_visualization()

    # 记录训练数据（用于保存npy）
    episodes_list = []
    total_rewards_list = []
    actor_losses_list = []
    critic_losses_list = []

    # 记录最佳奖励
    best_reward = -float('inf')
    total_rewards = []

    # 训练主循环
    print("===== 训练开始（仅食物奖励+碰撞惩罚）=====")
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0  # 初始奖励为0，无存活奖励
        trajectory = []
        uav_agent = env.agents[0]
        trajectory.append(uav_agent.state.p_pos.copy())
        actor_loss_ep = []
        critic_loss_ep = []

        for step in range(ep_length):
            # 采样动作
            ob = obs[0]
            action_tensor, _, _ = trainer.actor.sample(torch.FloatTensor(ob.astype(np.float32)))
            action = int(torch.clamp(action_tensor.round(), 0, max_action).item())
            actions = [action]

            # 执行动作
            next_obs, rewards, dones, info = env.step(actions)
            total_reward += rewards[0]  # 仅累加食物奖励/碰撞惩罚
            trajectory.append(uav_agent.state.p_pos.copy())

            # 实时可视化
            visualize_uav(ax, uav_agent, world, episode, step, total_reward, trajectory)

            # 存储经验
            action_oh = np.zeros(num_actions, dtype=np.float32)
            action_oh[action] = 1.0
            done_flag = dones[0] or scenario.done(uav_agent, world)
            replay_buffer.append({
                'states': obs[0],
                'actions': action_oh,
                'next_states': next_obs[0],
                'rewards': rewards[0],
                'dones': done_flag
            })

            # 训练更新
            if len(replay_buffer) >= batch_size:
                batch_samples = random.sample(replay_buffer, batch_size)
                batch = {
                    'states': np.stack([s['states'] for s in batch_samples]),
                    'actions': np.stack([s['actions'] for s in batch_samples]),
                    'next_states': np.stack([s['next_states'] for s in batch_samples]),
                    'rewards': np.array([[s['rewards']] for s in batch_samples]),
                    'dones': np.array([[s['dones']] for s in batch_samples])
                }
                loss_info = trainer.update(batch)
                actor_loss_ep.append(loss_info['actor_loss'])
                critic_loss_ep.append(loss_info['critic_loss'])

            # 碰撞立即终止回合
            if done_flag:
                print(f"Episode {episode} Step {step}: Collision detected! Terminate.")
                break

            obs = next_obs

        # 记录训练数据
        total_rewards.append(total_reward)
        episodes_list.append(episode)
        total_rewards_list.append(total_reward)
        actor_losses_list.append(np.mean(actor_loss_ep) if actor_loss_ep else 0)
        critic_losses_list.append(np.mean(critic_loss_ep) if critic_loss_ep else 0)

        # 保存日志和模型
        if episode % 50 == 0:
            avg_reward = np.mean(total_rewards[-50:]) if len(total_rewards) >= 50 else total_reward
            print(f"Episode {episode} | Total Reward: {total_reward:.1f} | Avg (50): {avg_reward:.1f}")
            # 保存当前模型
            torch.save({
                'actor': trainer.actor.state_dict(),
                'critic': trainer.critic.state_dict(),
                'episode': episode,
                'avg_reward': avg_reward
            }, os.path.join(save_dir, f'epc_actor_ep{episode}.pth'))
            # 更新最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save({
                    'actor': trainer.actor.state_dict(),
                    'critic': trainer.critic.state_dict(),
                    'best_reward': best_reward,
                    'best_episode': episode
                }, os.path.join(save_dir, 'epc_actor_best.pth'))

    # 保存npy文件用于画图
    train_logs = {
        "episodes": np.array(episodes_list, dtype=np.int32),
        "total_rewards": np.array(total_rewards_list, dtype=np.float32),
        "actor_losses": np.array(actor_losses_list, dtype=np.float32),
        "critic_losses": np.array(critic_losses_list, dtype=np.float32)
    }
    npy_save_path = os.path.join(save_dir, "train_logs_final_ctde_boost.npy")
    np.save(npy_save_path, train_logs)
    print(f"✅ EPC训练日志已保存为npy文件：{npy_save_path}")

    # 训练结束：保存最终模型
    final_model_path = os.path.join(save_dir, 'epc_actor_final.pth')
    torch.save({
        'actor': trainer.actor.state_dict(),
        'critic': trainer.critic.state_dict(),
        'total_episodes': num_episodes,
        'final_avg_reward': np.mean(total_rewards[-100:]),
        'best_reward': best_reward
    }, final_model_path)

    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, alpha=0.5, label='每回合奖励')
    if len(total_rewards) >= 50:
        moving_avg = np.convolve(total_rewards, np.ones(50) / 50, mode='valid')
        plt.plot(range(49, len(total_rewards)), moving_avg, 'r-', label='50回合滑动平均')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('EPC Training Reward Curve (仅食物奖励+碰撞惩罚)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'epc_training_curve.png'))
    plt.show()

    # 训练结束
    env.close()
    plt.ioff()
    plt.close()
    print(f"===== 训练完成 =====")
    print(f"📊 最后100回合平均奖励：{np.mean(total_rewards[-100:]):.1f}")
    print(f"🏆 最佳50回合平均奖励：{best_reward:.1f}")
    print(f"💾 最终模型保存至：{final_model_path}")