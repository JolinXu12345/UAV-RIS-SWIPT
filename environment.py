import numpy as np
import gym
from gym import spaces


class Environment:
    """UAV-RIS强化学习环境"""

    def __init__(self, config):
        """初始化环境"""
        # 配置参数
        self.bs_antennas = config.get('bs_antennas', 4)  # 基站天线数量
        self.ris_elements = config.get('ris_elements', 10)  # RIS元件数量
        self.fixed_action_dim = config.get('action_dim', 23)  # 固定动作维度

        # 计算理论上的动作维度 (但不使用它)
        # 3维UAV位置 + bs_antennas*2维波束成形 + ris_elements维相移
        self.theoretical_action_dim = 3 + self.bs_antennas * 2 + self.ris_elements

        # 状态空间维度
        self.state_dim = config.get('state_dim', 131)

        # 使用固定的动作维度，忽略动态计算的维度
        self.action_dim = self.fixed_action_dim

        # 动作和状态空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # 初始化其他必要的属性
        self.current_state = None
        self.steps = 0
        self.max_steps = config.get('max_steps', 100)
        self.bs_max_power = config.get('bs_max_power', 30)
        self.flying_period = config.get('flying_period', 100)
        self.power_splitting_ratio = config.get('power_splitting_ratio', 0.5)

        # 记录动作转换的详细信息，用于调试
        self.debug_mode = config.get('debug_mode', False)

        # 添加随机性因子，增加环境变化
        self.randomness_factor = config.get('randomness_factor', 0.2)

        # 奖励缩放因子 - 新增
        self.reward_scale = config.get('reward_scale', 0.1)  # 默认缩小10倍

        print(
            f"环境初始化完成 - 状态维度: {self.state_dim}, 理论动作维度: {self.theoretical_action_dim}, 使用的动作维度: {self.action_dim}")

    def reset(self):
        """重置环境"""
        # 重置环境状态，添加一些随机初始化以增加多样性
        self.current_state = np.random.normal(0, 0.1, self.state_dim).astype(np.float32)
        self.steps = 0

        # 返回初始状态
        return self.current_state

    def _adapt_action(self, action):
        """调整动作维度以适应环境预期的动作格式"""
        if self.debug_mode:
            print(f"原始动作维度: {action.shape}")

        # 确保动作维度正确
        if len(action) != self.action_dim:
            raise ValueError(f"动作维度不匹配: 期望{self.action_dim}，实际{len(action)}")

        # 将23维动作拆分为UAV位置、波束成形和相移
        uav_position = action[:3]  # 前3个维度是UAV位置

        # 处理波束成形 - 需要bs_antennas*2维度
        bf_dims = self.bs_antennas * 2
        if 3 + bf_dims <= len(action):
            beamforming = action[3:3 + bf_dims]
        else:
            # 如果不够，使用截断或填充
            available = max(0, len(action) - 3)
            beamforming = np.zeros(bf_dims)
            beamforming[:available] = action[3:3 + available]

        # 处理相移 - 需要ris_elements维度
        phase_dims = self.ris_elements
        if 3 + bf_dims + phase_dims <= len(action):
            phase_shifts = action[3 + bf_dims:3 + bf_dims + phase_dims]
        else:
            # 如果不够，使用截断或填充
            available = max(0, len(action) - 3 - bf_dims)
            phase_shifts = np.zeros(phase_dims)
            phase_shifts[:available] = action[3 + bf_dims:3 + bf_dims + available]

        # 返回调整后的动作组件
        return {
            'uav_position': uav_position,
            'beamforming': beamforming,
            'phase_shifts': phase_shifts
        }

    def step(self, action):
        """执行动作并返回下一个状态、奖励和结束标志"""
        # 步数增加
        self.steps += 1

        try:
            # 调整动作维度
            adapted_action = self._adapt_action(action)

            # 解包动作组件
            uav_position = adapted_action['uav_position']
            beamforming = adapted_action['beamforming']
            phase_shifts = adapted_action['phase_shifts']

            # 计算下一个状态，添加一些环境动态性
            # 每个状态有一定概率发生随机变化
            next_state = np.zeros(self.state_dim, dtype=np.float32)

            # 添加一些随机噪声使环境更动态
            env_noise = np.random.normal(0, self.randomness_factor * (1.0 - self.steps / self.max_steps),
                                         self.state_dim)
            next_state += env_noise

            # 添加一些周期性变化，模拟环境特征
            periodic_factor = np.sin(2 * np.pi * self.steps / self.flying_period)
            next_state += 0.1 * periodic_factor * np.ones(self.state_dim)

            # 计算奖励
            reward = self._calculate_reward(uav_position, beamforming, phase_shifts)

            # 计算是否完成
            done = self.steps >= self.max_steps

            # 保存当前状态
            self.current_state = next_state

            # 额外信息
            info = {
                'secrecy_rate': self._calculate_secrecy_rate(uav_position, beamforming, phase_shifts),
                'energy_efficiency': self._calculate_energy_efficiency(uav_position, beamforming, phase_shifts)
            }

            return next_state, reward, done, info

        except Exception as e:
            print(f"步骤执行错误: {e}")
            # 出错时返回零奖励并结束episode
            return self.current_state, 0.0, True, {'error': str(e)}

    def _calculate_reward(self, uav_position, beamforming, phase_shifts):
        """计算奖励函数"""
        # 计算保密率
        secrecy_rate = self._calculate_secrecy_rate(uav_position, beamforming, phase_shifts)

        # 计算能量效率
        energy_efficiency = self._calculate_energy_efficiency(uav_position, beamforming, phase_shifts)

        # 组合奖励，增加一些随机性模拟真实环境
        # 应用奖励缩放因子，控制奖励范围
        reward = (secrecy_rate + 0.5 * energy_efficiency) * self.reward_scale

        # 添加一些随机波动
        if np.random.random() < 0.2:  # 20%概率
            reward *= np.random.uniform(0.8, 1.2)

        # 限制最大奖励值
        reward = min(reward, 80.0)  # 设置一个上限

        return reward

    def _calculate_secrecy_rate(self, uav_position, beamforming, phase_shifts):
        """计算保密率 (示例实现)"""
        # 实现更动态的保密率计算
        try:
            # 引入非线性和随机性，但限制大小
            position_effect = np.sum(uav_position) + 0.05 * np.sin(np.sum(uav_position))
            beamforming_effect = 0.1 * np.sum(beamforming) + 0.02 * np.sum(beamforming ** 2)
            phase_effect = 0.15 * np.sum(phase_shifts) + 0.02 * np.cos(np.sum(phase_shifts))

            # 随机干扰因素
            interference = np.random.uniform(0.8, 1.2)

            secrecy_rate = interference * (position_effect + beamforming_effect + phase_effect)

            # 限制最大值
            secrecy_rate = min(max(0, secrecy_rate), 10.0)

            return secrecy_rate
        except Exception as e:
            print(f"计算保密率出错: {e}")
            return 0.0

    def _calculate_energy_efficiency(self, uav_position, beamforming, phase_shifts):
        """计算能量效率 (示例实现)"""
        try:
            # 更复杂和动态的能量效率计算
            # 动态功率消耗
            position_power = 0.1 * np.sum(uav_position ** 2)  # 位置相关功率
            beamforming_power = 0.5 * np.sum(beamforming ** 2)  # 波束功率
            phase_power = 0.05 * np.sum(np.abs(phase_shifts))  # 相移功率

            # 添加随机波动
            power_fluctuation = np.random.uniform(0.9, 1.1)
            total_power = max(0.5, (position_power + beamforming_power + phase_power) * power_fluctuation)

            # 保密率
            secrecy_rate = self._calculate_secrecy_rate(uav_position, beamforming, phase_shifts)

            # 能量效率，限制范围
            energy_efficiency = min(secrecy_rate / total_power, 5.0)

            return energy_efficiency
        except Exception as e:
            print(f"计算能量效率出错: {e}")
            return 0.0

    def close(self):
        """关闭环境并释放资源"""
        pass


# 绘图函数，使用渐变颜色和适当的线宽
def plot_rewards(rewards, save_path, title="Training Performance"):
    """绘制奖励曲线，使用移动平均但保留一定波动"""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 6))

    # 原始数据
    episodes = np.arange(1, len(rewards) + 1)

    # 移动平均，窗口选择适中以保留波动
    window_size = 15
    weights = np.ones(window_size) / window_size

    if len(rewards) > window_size:
        reward_avg = np.convolve(rewards, weights, 'valid')
        plt.plot(episodes[window_size - 1:], reward_avg, color='blue', linewidth=1.5)
    else:
        plt.plot(episodes, rewards, color='blue', linewidth=1.5)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# 如果直接运行此文件，则执行测试代码
if __name__ == "__main__":
    from td3 import TD3Agent
    import argparse
    import json
    import pickle
    import os
    import torch
    import matplotlib.pyplot as plt


    def parse_args():
        parser = argparse.ArgumentParser(description='Environment测试')
        parser.add_argument('--mode', type=str, default='train',
                            choices=['train', 'test', 'debug'],
                            help='运行模式')
        parser.add_argument('--config', type=str, default=None,
                            help='配置文件路径')
        return parser.parse_args()


    def main():
        args = parse_args()

        # 默认配置
        config = {
            'bs_antennas': 4,
            'ris_elements': 16,
            'action_dim': 23,  # 固定动作维度
            'state_dim': 131,
            'bs_max_power': 30,
            'num_episodes': 5000,  # 增加训练轮数
            'batch_size': 256,
            'save_dir': 'results/improved',
            'exploration_noise': 0.5,  # 增大初始探索噪声
            'exploration_decay': 0.995,  # 减缓衰减速率
            'exploration_min': 0.1,  # 提高最小探索噪声
            'max_steps': 100,
            'randomness_factor': 0.2,  # 环境随机性
            'reward_scale': 0.1,  # 奖励缩放因子，控制奖励大小
            # TD3特定参数
            'policy_noise': 0.3,  # 策略噪声
            'noise_clip': 0.6,  # 噪声裁剪
            'policy_freq': 1,  # 策略更新频率
            'tau': 0.01  # 软更新参数
        }

        # 如果有配置文件，加载它
        if args.config:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)

        print(f"运行模式: {args.mode}")
        print(f"配置: {config}")

        # 创建环境
        env = Environment(config)

        # 创建代理
        state_dim = env.state_dim
        action_dim = env.action_dim
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            policy_freq=config.get('policy_freq', 1),
            tau=config.get('tau', 0.01),
            policy_noise=config.get('policy_noise', 0.3),
            noise_clip=config.get('noise_clip', 0.6)
        )

        # 简单训练测试
        if args.mode == 'train':
            train_simple(env, agent, config)


    def train_simple(env, agent, config):
        """简单的训练循环"""
        save_dir = config.get('save_dir', 'results/improved')
        os.makedirs(save_dir, exist_ok=True)

        num_episodes = config.get('num_episodes', 1000)
        max_steps = config.get('max_steps', 100)
        batch_size = config.get('batch_size', 256)

        # 噪声参数
        exploration_base = config.get('exploration_noise', 0.5)
        exploration_decay = config.get('exploration_decay', 0.995)
        min_exploration = config.get('exploration_min', 0.1)

        rewards = []
        secrecy_rates = []
        energy_efficiency = []
        episode_steps = []

        print("开始训练...")
        print(f"设备: {agent.device}")

        for episode in range(1, num_episodes + 1):
            state = env.reset()
            done = False
            episode_reward = 0
            step = 0
            episode_secrecy_rate = []
            episode_energy_efficiency = []

            # 动态噪声策略
            if episode % 20 == 0:  # 每20个episodes临时增加探索
                noise = min(1.0, exploration_base * 1.5)  # 增大探索
            else:
                noise = max(min_exploration, exploration_base * (exploration_decay ** (episode / 50)))

            while not done and step < max_steps:
                step += 1

                # 选择动作，保持最小探索
                action = agent.select_action(state, noise=noise, min_noise=min_exploration)

                try:
                    # 执行动作
                    next_state, reward, done, info = env.step(action)

                    # 存储经验
                    agent.store_transition(state, action, reward, next_state, done)

                    # 更新网络
                    if len(agent.replay_buffer) > batch_size:
                        critic_loss, actor_loss = agent.update(batch_size)

                    # 更新状态和累计奖励
                    state = next_state
                    episode_reward += reward

                    # 收集指标
                    if 'secrecy_rate' in info:
                        episode_secrecy_rate.append(info['secrecy_rate'])
                    if 'energy_efficiency' in info:
                        episode_energy_efficiency.append(info['energy_efficiency'])

                except Exception as e:
                    print(f"步骤执行错误: {e}")
                    done = True

            # 记录episode数据
            rewards.append(episode_reward)
            episode_steps.append(step)

            # 计算平均指标
            avg_secrecy_rate = np.mean(episode_secrecy_rate) if episode_secrecy_rate else np.nan
            avg_energy_efficiency = np.mean(episode_energy_efficiency) if episode_energy_efficiency else np.nan
            secrecy_rates.append(avg_secrecy_rate)
            energy_efficiency.append(avg_energy_efficiency)

            # 打印训练进度
            print(f"Episode {episode}/{num_episodes} | Steps: {step} | Reward: {episode_reward:.3f} | "
                  f"Secrecy Rate: {avg_secrecy_rate:.3f} bps/Hz | "
                  f"Energy Efficiency: {avg_energy_efficiency:.3f} bps/Hz/W | "
                  f"Exploration: {noise:.3f}")

            # 定期保存训练指标
            if episode % 10 == 0 or episode == num_episodes:
                training_metrics = {
                    'rewards': rewards,
                    'secrecy_rates': secrecy_rates,
                    'energy_efficiency': energy_efficiency,
                    'episode_steps': episode_steps
                }
                with open(f"{save_dir}/training_metrics.pkl", 'wb') as f:
                    pickle.dump(training_metrics, f)

                # 绘制学习曲线
                plot_rewards(rewards, f"{save_dir}/reward_curve.png", "TD3 Training Rewards")
                plot_rewards(secrecy_rates, f"{save_dir}/secrecy_rate_curve.png", "Secrecy Rate")
                plot_rewards(energy_efficiency, f"{save_dir}/energy_efficiency_curve.png", "Energy Efficiency")

        # 保存最终模型
        try:
            agent.save(f"{save_dir}/model")
            print(f"训练完成，模型已保存到 {save_dir}")

            # 绘制最终学习曲线
            plot_rewards(rewards, f"{save_dir}/final_reward_curve.png", "TD3 Final Training Rewards")
        except Exception as e:
            print(f"保存最终模型或绘图错误: {e}")


    main()