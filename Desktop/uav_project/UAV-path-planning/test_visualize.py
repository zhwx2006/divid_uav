import torch
import numpy as np
import sys

sys.path.append('C:\\Users\\22895\\Desktop\\uav_project\\multiagent-particle-envs')

from multiagent.scenarios import load
from multiagent.environment import MultiAgentEnv
from epc_iclr import EPCActor, EPCTrainer

# 1. 加载环境
scenario = load("food_collection.py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

# 2. 加载训练好的模型
state_dim = env.observation_space[0].shape[0]
max_action = 4  # 离散动作范围0-4
action_dim = 1

trainer = EPCTrainer(state_dim, action_dim, max_action)
trainer.load_model("train_results/best_model.pth")  # 替换为你的模型路径

# 3. 运行测试并可视化
obs = env.reset()
total_reward = 0
for step in range(500):  # 运行500步
    # 模型预测动作（无探索，纯贪心）
    actions = []
    for i, ob in enumerate(obs):
        with torch.no_grad():  # 禁用梯度，提升速度
            action_tensor, _ = trainer.actor.sample(torch.FloatTensor(ob))
            action = int(torch.clamp(action_tensor.round(), 0, max_action).item())
        actions.append(action)

    # 环境步进，渲染画面
    obs, rewards, dones, _ = env.step(actions)
    env.render(mode='human')  # 显示Pygame窗口
    total_reward += sum(rewards)

    if any(dones):
        break

print(f"测试总奖励: {total_reward:.1f}")
env.close()