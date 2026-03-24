import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# ========== 无人机动力学配置（保持不变） ==========
class UAVDynamics:
    def __init__(self):
        self.max_speed = 0.15  # 最大速度（m/s）
        self.max_acc = 0.05    # 最大加速度（m/s²）
        self.dt = 0.1          # 时间步长（s）

    def update_state(self, current_pos, current_vel, action):
        action_map = {
            0: np.array([0.0, self.max_acc]),    # 上
            1: np.array([0.0, -self.max_acc]),   # 下
            2: np.array([-self.max_acc, 0.0]),   # 左
            3: np.array([self.max_acc, 0.0])     # 右
        }
        acc = action_map[action]
        new_vel = np.clip(current_vel + acc * self.dt, -self.max_speed, self.max_speed)
        new_pos = current_pos + new_vel * self.dt + 0.5 * acc * (self.dt ** 2)
        return new_pos, new_vel


# ========== Actor 网络（严格匹配表中配置） ==========
class Actor(nn.Module):
    def __init__(self, state_dim=7, action_dim=2):
        super(Actor, self).__init__()
        # 隐藏层：(256, 256)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        # 权重初始化：(0, 0.1)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)

    def forward(self, state):
        x = F.relu(self.fc1(state))   # 隐藏层激活：relu
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))   # 输出层激活：tanh
        return x

    def sample(self, state):
        logits = self.forward(state)
        # 适配连续动作输出（表中Output:2）
        action_dist = torch.distributions.Normal(logits, torch.ones_like(logits)*0.1)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy


# ========== Critic 网络（严格匹配表中配置） ==========
class Critic(nn.Module):
    def __init__(self, input_dim=9):  # 单智能体时：7*1 + 2 = 9
        super(Critic, self).__init__()
        # 隐藏层：(256, 256)
        self.fc1_1 = nn.Linear(input_dim, 256)
        self.fc1_2 = nn.Linear(256, 256)
        self.fc1_3 = nn.Linear(256, 1)

        self.fc2_1 = nn.Linear(input_dim, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.fc2_3 = nn.Linear(256, 1)
        # 权重初始化：(0, 0.1)
        nn.init.normal_(self.fc1_1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc1_2.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc1_3.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2_1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2_2.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2_3.weight, mean=0, std=0.1)

    def forward(self, global_state_action):
        x1 = F.relu(self.fc1_1(global_state_action))  # 激活：relu
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.fc1_3(x1)

        x2 = F.relu(self.fc2_1(global_state_action))
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.fc2_3(x2)
        return q1, q2

    def q1_forward(self, global_state_action):
        x = F.relu(self.fc1_1(global_state_action))
        x = F.relu(self.fc1_2(x))
        q1 = self.fc1_3(x)
        return q1


# ========== MASAC Trainer（超参数完全对齐表中配置） ==========
class EPCTrainer:
    def __init__(self, num_agents=1, state_dim=7, action_dim=2,
                 actor_lr=1e-3, critic_lr=3e-3, entropy_lr=3e-4,
                 gamma=0.9, tau=1e-2):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  # Soft update rate: 1e-2

        # Actor 网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic 输入维度：7*N + 2（单智能体 N=1 → 9）
        self.critic_input_dim = 7 * num_agents + 2
        self.critic = Critic(self.critic_input_dim)
        self.critic_target = Critic(self.critic_input_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器：Adam，学习率对齐表中
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.entropy_optimizer = optim.Adam([torch.tensor(0.0)], lr=entropy_lr)
        self.entropy_coeff = torch.tensor(0.01, requires_grad=True)  # 可学习熵系数

    def update(self, batch):
        # 数据维度：[batch_size, state_dim] / [batch_size, action_dim]
        states = torch.FloatTensor(batch['states'])
        next_states = torch.FloatTensor(batch['next_states'])
        actions = torch.FloatTensor(batch['actions'])
        rewards = torch.FloatTensor(batch['rewards'])
        dones = torch.FloatTensor(batch['dones'])

        # 全局输入拼接：state + action
        global_input = torch.cat([states, actions], dim=1)

        # 目标 Q 值计算
        with torch.no_grad():
            next_actions, _, _ = self.actor_target.sample(next_states)
            global_next_input = torch.cat([next_states, next_actions], dim=1)
            target_q1, target_q2 = self.critic_target(global_next_input)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Critic 更新
        current_q1, current_q2 = self.critic(global_input)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # Actor 更新
        pred_actions, pred_log_probs, pred_entropies = self.actor.sample(states)
        global_pred_input = torch.cat([states, pred_actions], dim=1)
        q1, q2 = self.critic(global_pred_input)
        q = torch.min(q1, q2)
        actor_loss = - (q + self.entropy_coeff * pred_entropies).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # 熵系数更新
        entropy_loss = (-self.entropy_coeff * (pred_entropies - self.action_dim).detach()).mean()
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_coeff': self.entropy_coeff.item()
        }