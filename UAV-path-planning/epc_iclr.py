import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# ========== CTDE 核心网络（动态维度适配） ==========
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample(self, state):
        logits = self.forward(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()  # 0~action_dim-1
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, input_dim):  # 动态接收输入维度
        super(Critic, self).__init__()
        # 输入维度动态适配，不再硬编码
        self.fc1_1 = nn.Linear(input_dim, 512)
        self.fc1_2 = nn.Linear(512, 256)
        self.fc1_3 = nn.Linear(256, 1)

        self.fc2_1 = nn.Linear(input_dim, 512)
        self.fc2_2 = nn.Linear(512, 256)
        self.fc2_3 = nn.Linear(256, 1)

    def forward(self, global_state_action):
        # 直接接收拼接后的全局状态+动作，避免维度拼接错误
        x = F.relu(self.fc1_1(global_state_action))
        x = F.relu(self.fc1_2(x))
        q1 = self.fc1_3(x)

        x = F.relu(self.fc2_1(global_state_action))
        x = F.relu(self.fc2_2(x))
        q2 = self.fc2_3(x)
        return q1, q2

    def q1_forward(self, global_state_action):
        x = F.relu(self.fc1_1(global_state_action))
        x = F.relu(self.fc1_2(x))
        q1 = self.fc1_3(x)
        return q1


# ========== CTDE Trainer（最终稳定版） ==========
class EPCTrainer:
    def __init__(self, num_agents, state_dim, action_dim, lr=8e-6, gamma=0.99, entropy_coeff=0.0002):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

        # 初始化Actor
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # 计算Critic输入维度（全局状态+全局动作）
        self.critic_input_dim = num_agents * state_dim + num_agents * action_dim
        # 初始化Critic（动态维度）
        self.critic = Critic(self.critic_input_dim)
        self.critic_target = Critic(self.critic_input_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.tau = 0.007  # 微调：0.008→0.007，目标网络更新更慢

    def update(self, batch):
        # 1. 提取数据并转换为tensor
        states = torch.FloatTensor(batch['states'])  # [batch, num_agents, state_dim]
        next_states = torch.FloatTensor(batch['next_states'])  # [batch, num_agents, state_dim]
        actions = torch.FloatTensor(batch['actions'])  # [batch, num_agents, action_dim]
        rewards = torch.FloatTensor(batch['rewards'])  # [batch, num_agents, 1]
        dones = torch.FloatTensor(batch['dones'])  # [batch, num_agents, 1]

        # 2. 构造全局输入（状态+动作拼接）
        batch_size = states.shape[0]
        # 展平状态和动作
        flat_states = states.view(batch_size, -1)  # [batch, num_agents*state_dim]
        flat_next_states = next_states.view(batch_size, -1)  # [batch, num_agents*state_dim]
        flat_actions = actions.view(batch_size, -1)  # [batch, num_agents*action_dim]

        # 拼接全局状态+动作
        global_input = torch.cat([flat_states, flat_actions], dim=1)  # [batch, critic_input_dim]

        # 3. 计算目标Q值（使用target网络）
        with torch.no_grad():
            # 预测下一个动作
            next_actions = []
            next_log_probs = []
            for i in range(self.num_agents):
                a, lp, _ = self.actor_target.sample(next_states[:, i, :])
                # 转换为one-hot编码
                a_onehot = F.one_hot(a.long(), num_classes=self.action_dim).float()
                next_actions.append(a_onehot.unsqueeze(1))
                next_log_probs.append(lp.unsqueeze(1))

            next_actions = torch.cat(next_actions, dim=1)  # [batch, num_agents, action_dim]
            flat_next_actions = next_actions.view(batch_size, -1)  # [batch, num_agents*action_dim]
            # 拼接下一个全局状态+动作
            global_next_input = torch.cat([flat_next_states, flat_next_actions], dim=1)

            # 计算target Q值
            target_q1, target_q2 = self.critic_target(global_next_input)
            target_q = torch.min(target_q1, target_q2)
            # 奖励归一化+终止标志处理
            reward = rewards.mean(dim=1, keepdim=True)  # [batch, 1]
            done = dones.mean(dim=1, keepdim=True)  # [batch, 1]
            target_q = reward + (1 - done) * self.gamma * target_q

        # 4. 更新Critic
        current_q1, current_q2 = self.critic(global_input)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.15)
        self.critic_optimizer.step()

        # 5. 更新Actor（最大化Q值+熵正则）
        pred_actions = []
        pred_log_probs = []
        pred_entropies = []
        for i in range(self.num_agents):
            a, lp, ent = self.actor.sample(states[:, i, :])
            # 转换为one-hot编码
            a_onehot = F.one_hot(a.long(), num_classes=self.action_dim).float()
            pred_actions.append(a_onehot.unsqueeze(1))
            pred_log_probs.append(lp.unsqueeze(1))
            pred_entropies.append(ent.unsqueeze(1))

        pred_actions = torch.cat(pred_actions, dim=1)  # [batch, num_agents, action_dim]
        flat_pred_actions = pred_actions.view(batch_size, -1)  # [batch, num_agents*action_dim]
        # 拼接预测动作的全局输入
        global_pred_input = torch.cat([flat_states, flat_pred_actions], dim=1)

        # 计算Actor损失
        q1, q2 = self.critic(global_pred_input)
        q = torch.min(q1, q2)
        pred_log_probs = torch.cat(pred_log_probs, dim=1)
        pred_entropies = torch.cat(pred_entropies, dim=1)

        # 调整损失计算，确保Actor损失为正
        actor_loss = - (q + self.entropy_coeff * pred_entropies.mean(dim=1, keepdim=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.15)
        self.actor_optimizer.step()

        # 6. 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }