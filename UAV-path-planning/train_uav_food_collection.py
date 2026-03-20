# 环境配置
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

# Win11兼容 + 屏蔽警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings

warnings.filterwarnings("ignore")

# ========== 关键修正：匹配实际scenarios路径 ==========
MPE_PATH = "C:\\Users\\22895\\Desktop\\uav_project\\multiagent-particle-envs"
SCENARIOS_PATH = os.path.join(MPE_PATH, "multiagent", "scenarios")
sys.path.append(MPE_PATH)
sys.path.append(SCENARIOS_PATH)
os.chdir('C:\\Users\\22895\\Desktop\\uav_project\\UAV-path-planning')

# 手动导入food_collection模块
import importlib

try:
    food_collection = importlib.import_module("food_collection")
    Scenario = food_collection.Scenario
    print(f"✅ 成功从 {SCENARIOS_PATH} 导入Scenario类")
except Exception as e:
    print(f"❌ 导入失败：{e}")
    print(f"🔍 检查路径：{SCENARIOS_PATH}/food_collection.py 是否存在")
    sys.exit(1)

# 导入MPE核心模块
from multiagent.environment import MultiAgentEnv
from multiagent.core import World


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

    # 绘制轨迹和无人机（中心点）
    if len(trajectory) > 1:
        ax.plot([p[0] for p in trajectory], [p[1] for p in trajectory], 'b-', alpha=0.5, label='Trajectory')
    ax.scatter(agent.state.p_pos[0], agent.state.p_pos[1], color='red', s=50, marker='^', label='UAV (Center)')
    ax.legend(loc='upper right')
    plt.draw()
    plt.pause(0.001)


# ========== 网络定义 ==========
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# ========== 训练器 ==========
class DDPG:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau

        # 演员网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)

        # 评论家网络
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)

        # 同步目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.actor(state)
        # 动态温度系数：训练后期降低探索性
        temperature = max(0.05, 0.1 - (self.training_step / 100000)) if hasattr(self, 'training_step') else 0.1
        self.training_step = getattr(self, 'training_step', 0) + 1
        action = torch.distributions.Categorical(logits=logits / temperature).sample()
        return action.item()

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return 0, 0

        # 采样数据
        batch = random.sample(replay_buffer, batch_size)
        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor(np.array([x[1] for x in batch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
        dones = torch.FloatTensor(np.array([x[4] for x in batch])).unsqueeze(1)

        # 独热编码动作
        actions_onehot = F.one_hot(actions, num_classes=self.actor.fc3.out_features).float()

        # 更新评论家
        next_actions = self.actor_target(next_states)
        next_actions_onehot = F.one_hot(torch.argmax(next_actions, 1), num_classes=self.actor.fc3.out_features).float()
        target_q = self.critic_target(next_states, next_actions_onehot)
        target_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.critic(states, actions_onehot)

        critic_loss = F.mse_loss(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 更新演员
        actor_actions = self.actor(states)
        actor_actions_onehot = F.one_hot(torch.argmax(actor_actions, 1),
                                         num_classes=self.actor.fc3.out_features).float()
        actor_loss = -self.critic(states, actor_actions_onehot).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()


# ========== 主训练流程 ==========
if __name__ == "__main__":
    # 配置参数
    num_episodes = 2000
    ep_length = 300
    batch_size = 128
    lr = 2e-5
    gamma = 0.95
    save_dir = "./uav_epc_results"
    os.makedirs(save_dir, exist_ok=True)

    # 测试路径
    test_dir = "./train_results"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test.txt", "w") as f:
        f.write("路径可写")
    print(f"✅ 测试文件已生成：{test_dir}/test.txt")

    # 实例化Scenario
    scenario = Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        done_callback=None,
        shared_viewer=False
    )

    # 环境维度
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    print(f"===== 训练开始 =====")
    print(f"📊 状态维度：{state_dim}, 动作维度：{action_dim}")
    print(f"🚀 无人机速度：{world.agents[0].max_speed}")

    # 初始化训练器和回放池
    agent = DDPG(state_dim, action_dim, lr=lr, gamma=gamma)
    replay_buffer = []
    fig, ax = init_visualization()

    # 记录训练数据（用于保存npy）
    episodes_list = []
    total_rewards_list = []
    actor_losses_list = []
    critic_losses_list = []

    # 记录最佳奖励
    best_reward = -float('inf')
    total_rewards = []

    # 训练循环
    for episode in range(num_episodes):
        obs = env.reset()
        world.collision_terminate = False
        total_reward = 0  # 初始奖励为0，无任何基础奖励
        actor_losses = []
        critic_losses = []
        trajectory = []

        for step in range(ep_length):
            # 获取无人机状态和位置
            uav_obs = obs[0]
            uav_agent = env.world.agents[0]
            trajectory.append(uav_agent.state.p_pos.copy())

            # 选择动作并执行
            action = agent.select_action(uav_obs)
            next_obs, rewards, dones, _ = env.step([action])

            # 手动判断终止
            done = world.collision_terminate or step == ep_length - 1
            total_reward += rewards[0]  # 仅累加食物奖励/碰撞惩罚，无其他

            # 存储经验
            replay_buffer.append([uav_obs, action, rewards[0], next_obs[0], done])
            replay_buffer = replay_buffer[-200000:]

            # 更新网络
            a_loss, c_loss = agent.update(replay_buffer, batch_size)
            if a_loss > 0:
                actor_losses.append(a_loss)
                critic_losses.append(c_loss)

            # 可视化
            visualize_uav(ax, uav_agent, world, episode, step, total_reward, trajectory)

            # 终止判断
            if done:
                terminate_reason = "Collision" if world.collision_terminate else "Max Step"
                print(
                    f"📌 Episode {episode:4d} | Step {step:3d} | Reward: {total_reward:6.1f} | Terminate: {terminate_reason}")
                break

            obs = next_obs

        # 记录训练数据
        total_rewards.append(total_reward)
        episodes_list.append(episode)
        total_rewards_list.append(total_reward)
        actor_losses_list.append(np.mean(actor_losses) if actor_losses else 0)
        critic_losses_list.append(np.mean(critic_losses) if critic_losses else 0)

        # 每50回合保存模型+打印汇总
        if episode % 50 == 0:
            avg_a_loss = np.mean(actor_losses) if actor_losses else 0
            avg_c_loss = np.mean(critic_losses) if critic_losses else 0
            avg_reward = np.mean(total_rewards[-50:]) if len(total_rewards) >= 50 else total_reward
            print(f"\n===== Episode {episode} 训练汇总 =====")
            print(f"🏆 总奖励：{total_reward:6.1f} | 近50回合平均：{avg_reward:6.1f}")
            print(f"🧠 Actor Loss：{avg_a_loss:.4f} | Critic Loss：{avg_c_loss:.4f}")

            # 保存模型
            model_path = f"{save_dir}/uav_ep{episode}.pth"
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'episode': episode,
                'total_reward': total_reward,
                'avg_reward_50': avg_reward
            }, model_path)
            print(f"💾 模型已保存至：{model_path}\n")

            # 更新最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model_path = f"{save_dir}/uav_best.pth"
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'episode': episode,
                    'best_reward': best_reward
                }, best_model_path)
                print(f"🏆 最佳模型已更新（奖励：{best_reward:.1f}）：{best_model_path}")

    # 保存npy文件用于画图
    train_logs = {
        "episodes": np.array(episodes_list, dtype=np.int32),
        "total_rewards": np.array(total_rewards_list, dtype=np.float32),
        "actor_losses": np.array(actor_losses_list, dtype=np.float32),
        "critic_losses": np.array(critic_losses_list, dtype=np.float32)
    }
    npy_save_path = os.path.join(save_dir, "train_logs_final_ctde_boost.npy")
    np.save(npy_save_path, train_logs)
    print(f"✅ 训练日志已保存为npy文件：{npy_save_path}")

    # 训练结束：保存最终模型
    final_model_path = f"{save_dir}/uav_final.pth"
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'total_episodes': num_episodes,
        'avg_final_reward': np.mean(total_rewards[-100:]),
        'best_reward': best_reward
    }, final_model_path)
    print(f"\n💾 最终模型已保存至：{final_model_path}")
    print(f"🏆 训练期间最佳平均奖励：{best_reward:.1f}")
    print(f"📊 最后100回合平均奖励：{np.mean(total_rewards[-100:]):.1f}")

    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, alpha=0.5, label='每回合奖励')
    # 滑动平均
    window_size = 50
    if len(total_rewards) >= window_size:
        moving_avg = np.convolve(total_rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(total_rewards)), moving_avg, 'r-', label=f'{window_size}回合滑动平均')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curve (仅食物奖励+碰撞惩罚)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_reward_curve.png")
    plt.show()

    # 训练结束
    env.close()
    plt.ioff()
    plt.show()
    print(f"\n🎉 训练完成！所有模型和日志保存至：{save_dir}")