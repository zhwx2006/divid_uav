import matplotlib.pyplot as plt
import numpy as np
import os

# ====================== 核心修复：解决中文乱码 + 数据渲染问题 ======================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 1. 加载训练日志（适配2000回合日志路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 注意：训练完成后最终日志是 train_logs_final_ctde_boost.npy，训练中是 train_logs.npy
log_path = os.path.join(current_dir, "uav_epc_results", "train_logs_final_ctde_boost.npy")

# 检查日志是否存在
if not os.path.exists(log_path):
    # 兜底：加载训练中的临时日志
    log_path = os.path.join(current_dir, "uav_epc_results", "train_logs_final_ctde_boost.npy")
    if not os.path.exists(log_path):
        print(f"❌ 未找到日志文件！路径：{log_path}")
        input("按任意键退出...")
        exit()

# 2. 核心修复：强制加载并转换为数值类型
train_logs = np.load(log_path, allow_pickle=True).item()
# 强制转换为numpy数组（解决object类型导致的渲染失败）
episodes = np.array(train_logs['episodes'], dtype=np.int32)
total_rewards = np.array(train_logs['total_rewards'], dtype=np.float32)
actor_losses = np.array(train_logs['actor_losses'], dtype=np.float32)
critic_losses = np.array(train_logs['critic_losses'], dtype=np.float32)

print(f"✅ 成功加载日志：共 {len(episodes)} 个训练回合")
print(f"✅ 回合范围：{episodes.min()} ~ {episodes.max()}")
print(f"✅ 奖励范围：{total_rewards.min():.1f} ~ {total_rewards.max():.1f}")

# 3. 绘制曲线（自动适配坐标轴范围）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100)

# -------------------------- 奖励曲线（核心修复：自动适配坐标轴） --------------------------
ax1.plot(episodes, total_rewards, color='#2E8B57', linewidth=1.5, alpha=0.8, label='每回合总奖励')
# 可选：添加10回合平滑曲线（2000回合更易看趋势）
if len(episodes) >= 10:
    window_size = 10
    smoothed_rewards = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')
    smoothed_episodes = episodes[window_size-1:]
    ax1.plot(smoothed_episodes, smoothed_rewards, color='#FF6347', linewidth=2, label=f'{window_size}回合平滑奖励')

# 自动适配坐标轴（核心：删除手动设置范围，让Matplotlib自动计算）
ax1.set_xlabel('训练回合', fontsize=12, fontweight='bold')
ax1.set_ylabel('总奖励值', fontsize=12, fontweight='bold')
ax1.set_title('无人机Food Collection训练奖励趋势（2000回合）', fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper right')
ax1.tick_params(axis='both', labelsize=10)

# -------------------------- 损失曲线（核心修复：适配负损失） --------------------------
ax2.plot(episodes, actor_losses, color='#4169E1', linewidth=1.5, alpha=0.8, label='Actor损失')
ax2.plot(episodes, critic_losses, color='#DC143C', linewidth=1.5, alpha=0.8, label='Critic损失')

# 自动适配坐标轴（兼容Actor负损失）
ax2.set_xlabel('训练回合', fontsize=12, fontweight='bold')
ax2.set_ylabel('损失值', fontsize=12, fontweight='bold')
ax2.set_title('Actor/Critic网络损失趋势（2000回合）', fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='upper right')
ax2.tick_params(axis='both', labelsize=10)

# 4. 调整布局并保存（避免文字截断）
plt.tight_layout(pad=3.0)
save_path = os.path.join(current_dir, "uav_epc_results", "training_curves_2000ep.png")
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
print(f"✅ 2000回合训练曲线已保存：{save_path}")

# 显示曲线
plt.show()

# 输出关键指标
print("\n📊 2000回合训练关键指标：")
print(f"   - 最高奖励：{total_rewards.max():.1f}（回合 {episodes[np.argmax(total_rewards)]}）")
print(f"   - 最低奖励：{total_rewards.min():.1f}（回合 {episodes[np.argmin(total_rewards)]}）")
print(f"   - 最后100回合平均奖励：{total_rewards[-100:].mean():.1f}")
print(f"   - 最终Critic损失：{critic_losses[-1]:.4f}")
input("按任意键关闭窗口...")