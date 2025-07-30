import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from scipy.constants import c  # speed of light
import os
from data_manager import DataManager


class UAVRISEnvironment(gym.Env):
    """
    UAV-RIS辅助SWIPT系统的物理层安全环境

    环境支持优化:
    - BS的发射波束成形
    - UAV的位置
    - RIS的相移矩阵
    - 功率分配比例

    目标: 最大化系统保密率同时保证能量收集需求
    """

    def __init__(self, config=None):
        super(UAVRISEnvironment, self).__init__()

        # 默认配置参数
        self.config = {
            'bs_antennas': 4,  # 基站天线数量
            'ris_elements': 16,  # RIS反射元件数量
            'max_uav_height': 100,  # UAV最大飞行高度 (m)
            'min_uav_height': 50,  # UAV最小飞行高度 (m)
            'area_size': 1000,  # 区域大小 (m x m)
            'carrier_frequency': 2.4e9,  # 载波频率 (Hz)
            'bs_max_power': 30,  # 最大基站功率 (dBm)
            'noise_power': -110,  # 噪声功率 (dBm)
            'bs_position': [0, 0, 25],  # 基站位置 [x, y, z] (m)
            'bob_position': [500, 500, 1.5],  # 合法用户位置 (m)
            'eve_position': [400, 400, 1.5],  # 窃听者位置 (m)
            'min_rate': 1.0,  # 最小数据速率要求 (bps/Hz)
            'min_secrecy_rate': 0.5,  # 最小保密率要求 (bps/Hz)
            'min_harvested_energy': 0.1,  # 最小能量收集要求 (W)
            'power_splitting_min': 0.1,  # 最小功率分配比例
            'power_splitting_max': 0.9,  # 最大功率分配比例
            'flying_period': 100,  # 飞行周期
            'max_steps': 100,  # 每个episode的最大步数
            'energy_weight': 0.3,  # 奖励函数中能量收集的权重
            'secrecy_weight': 0.7,  # 奖励函数中保密率的权重
            'energy_harvesting_efficiency': 0.5,  # 能量收集效率
            'randomness_factor': 0.1,  # 环境随机性因子
            'rice_factor': 3.0,  # 莱斯衰落因子
            'path_loss_exponent': 2.0,  # 路径损耗指数
            'los_coefficient': 0.1,  # LoS概率系数
            'nlos_coefficient': 21,  # NLoS概率系数
            'debug_mode': False,  # 调试模式标志
            'store_path': './results',  # 存储路径
        }

        # 更新配置
        if config:
            self.config.update(config)

        # 创建数据管理器
        self.data_manager = DataManager(self.config['store_path'])

        # 设置动作空间维度
        bs_bf_dim = 2 * self.config['bs_antennas']  # 波束成形复数向量
        an_bf_dim = 2 * self.config['bs_antennas']  # 人工噪声波束成形复数向量
        ris_phase_dim = self.config['ris_elements']  # RIS相移向量
        uav_position_dim = 3  # UAV的3D位置 [x, y, z]
        power_splitting_dim = 1  # 功率分配比例

        self.action_dim = bs_bf_dim + an_bf_dim + ris_phase_dim + uav_position_dim + power_splitting_dim

        # 设置观察空间维度
        # B-U信道、U-B信道、U-E信道等
        bs_u_channel_dim = 2 * self.config['bs_antennas'] * self.config['ris_elements']  # 复数信道
        u_bob_channel_dim = 2 * self.config['ris_elements']  # 复数信道
        u_eve_channel_dim = 2 * self.config['ris_elements']  # 复数信道
        uav_position_dim = 3  # UAV当前位置
        uav_energy_dim = 1  # UAV当前能量水平

        self.state_dim = bs_u_channel_dim + u_bob_channel_dim + u_eve_channel_dim + uav_position_dim + uav_energy_dim

        # 定义动作空间和观察空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # 初始化环境状态
        self.current_state = None
        self.uav_position = None
        self.uav_energy = None
        self.bs_u_channel = None
        self.u_bob_channel = None
        self.u_eve_channel = None
        self.steps = 0

        # 初始化性能指标记录
        self.secrecy_rates = []
        self.energy_efficiency = []

        # 初始化最佳结果记录
        self.best_secrecy_rate = 0
        self.best_energy_harvested = 0
        self.best_energy_efficiency = 0
        self.best_config = None

        # 初始化用于绘图的数据
        self.power_allocation_data = []
        self.transmit_power_data = []

        print(f"UAV-RIS环境初始化完成 - 状态维度: {self.state_dim}, 动作维度: {self.action_dim}")

    def reset(self):
        """重置环境到初始状态，返回初始观察"""
        self.steps = 0

        # 随机初始化UAV位置
        x = np.random.uniform(0, self.config['area_size'])
        y = np.random.uniform(0, self.config['area_size'])
        z = np.random.uniform(self.config['min_uav_height'], self.config['max_uav_height'])
        self.uav_position = np.array([x, y, z])

        # 初始化UAV能量水平 (满电量)
        self.uav_energy = 1.0

        # 生成初始信道状态信息
        self.bs_u_channel = self._generate_bs_u_channel()
        self.u_bob_channel = self._generate_u_bob_channel()
        self.u_eve_channel = self._generate_u_eve_channel()

        # 构建状态向量
        self.current_state = self._build_state()

        return self.current_state

    def step(self, action):
        """执行动作并返回下一个状态、奖励和终止标志"""
        # 步数增加
        self.steps += 1

        try:
            # 动作预处理 (将[-1,1]范围映射到实际物理量)
            processed_action = self._preprocess_action(action)

            # 解析动作
            bs_beamforming = processed_action['bs_beamforming']
            an_beamforming = processed_action['an_beamforming']
            ris_phase_shift = processed_action['ris_phase_shift']
            uav_position_delta = processed_action['uav_position_delta']
            power_splitting_ratio = processed_action['power_splitting_ratio']

            # 更新UAV位置
            self.uav_position += uav_position_delta

            # 确保UAV位置在有效范围内
            self.uav_position[0] = np.clip(self.uav_position[0], 0, self.config['area_size'])
            self.uav_position[1] = np.clip(self.uav_position[1], 0, self.config['area_size'])
            self.uav_position[2] = np.clip(self.uav_position[2],
                                           self.config['min_uav_height'],
                                           self.config['max_uav_height'])

            # 更新信道状态
            self.bs_u_channel = self._generate_bs_u_channel()
            self.u_bob_channel = self._generate_u_bob_channel()
            self.u_eve_channel = self._generate_u_eve_channel()

            # 计算性能指标
            secrecy_rate = self._calculate_secrecy_rate(bs_beamforming, an_beamforming,
                                                        ris_phase_shift, power_splitting_ratio)
            harvested_energy = self._calculate_harvested_energy(bs_beamforming, an_beamforming,
                                                                ris_phase_shift, power_splitting_ratio)
            energy_efficiency = self._calculate_energy_efficiency(secrecy_rate, harvested_energy)

            # 更新UAV能量水平
            self.uav_energy = min(1.0, self.uav_energy - 0.01 + harvested_energy * 0.1)

            # 计算奖励
            reward = self._calculate_reward(secrecy_rate, harvested_energy)

            # 记录性能指标
            self.secrecy_rates.append(secrecy_rate)
            self.energy_efficiency.append(energy_efficiency)

            # 更新最佳结果
            if secrecy_rate > self.best_secrecy_rate:
                self.best_secrecy_rate = secrecy_rate
                self.best_config = {
                    'secrecy_rate': secrecy_rate,
                    'harvested_energy': harvested_energy,
                    'energy_efficiency': energy_efficiency,
                    'uav_position': self.uav_position.copy(),
                    'power_splitting_ratio': power_splitting_ratio
                }

            if harvested_energy > self.best_energy_harvested:
                self.best_energy_harvested = harvested_energy

            if energy_efficiency > self.best_energy_efficiency:
                self.best_energy_efficiency = energy_efficiency

            # 记录功率分配数据
            self.power_allocation_data.append(power_splitting_ratio)

            # 计算发射功率 (dBm)
            P_tx = self.config['bs_max_power']
            self.transmit_power_data.append(P_tx)

            # 构建新状态
            next_state = self._build_state()
            self.current_state = next_state

            # 判断是否结束
            done = self.steps >= self.config['max_steps'] or self.uav_energy <= 0

            # 额外信息
            info = {
                'secrecy_rate': secrecy_rate,
                'harvested_energy': harvested_energy,
                'energy_efficiency': energy_efficiency,
                'uav_position': self.uav_position.copy(),
                'uav_energy': self.uav_energy
            }

            return next_state, reward, done, info

        except Exception as e:
            print(f"步骤执行错误: {e}")
            return self.current_state, -10.0, True, {'error': str(e)}

    def _calculate_reward(self, secrecy_rate, harvested_energy):
        """计算奖励函数 - 基于保密率和能量收集的加权组合"""
        # 获取配置中的权重
        secrecy_weight = self.config['secrecy_weight']
        energy_weight = self.config['energy_weight']

        # 获取约束条件的最小值
        min_secrecy_rate = self.config['min_secrecy_rate']
        min_harvested_energy = self.config['min_harvested_energy']

        # 计算保密率项的奖励 (高于要求值时奖励更高)
        if secrecy_rate >= min_secrecy_rate:
            secrecy_reward = secrecy_rate
        else:
            # 未达到最低要求时惩罚
            secrecy_reward = secrecy_rate - 2 * (min_secrecy_rate - secrecy_rate)

        # 计算能量收集项的奖励 (高于要求值时奖励更高)
        if harvested_energy >= min_harvested_energy:
            energy_reward = harvested_energy
        else:
            # 未达到最低要求时惩罚
            energy_reward = harvested_energy - 2 * (min_harvested_energy - harvested_energy)

        # 合并奖励 (使用指定权重)
        reward = secrecy_weight * secrecy_reward + energy_weight * energy_reward

        return reward

    def _preprocess_action(self, action):
        """将神经网络输出的动作转换为物理意义的控制量"""
        # 提取各个部分的动作
        bs_antennas = self.config['bs_antennas']
        ris_elements = self.config['ris_elements']

        # 波束成形向量 (复数)
        bs_bf_dim = 2 * bs_antennas
        bs_beamforming_raw = action[:bs_bf_dim]

        # 人工噪声波束成形向量 (复数)
        an_bf_dim = 2 * bs_antennas
        an_beamforming_raw = action[bs_bf_dim:bs_bf_dim + an_bf_dim]

        # RIS相移向量
        ris_phase_raw = action[bs_bf_dim + an_bf_dim:bs_bf_dim + an_bf_dim + ris_elements]

        # UAV位置调整
        uav_pos_raw = action[bs_bf_dim + an_bf_dim + ris_elements:bs_bf_dim + an_bf_dim + ris_elements + 3]

        # 功率分配比例
        power_splitting_raw = action[-1]

        # 转换波束成形向量为复数形式
        bs_beamforming = np.zeros(bs_antennas, dtype=np.complex128)
        for i in range(bs_antennas):
            real_part = bs_beamforming_raw[2 * i]
            imag_part = bs_beamforming_raw[2 * i + 1]
            amplitude = np.sqrt(real_part ** 2 + imag_part ** 2)
            if amplitude > 0:
                bs_beamforming[i] = complex(real_part / amplitude, imag_part / amplitude)
            else:
                bs_beamforming[i] = complex(1, 0)

        # 标准化波束成形向量
        norm = np.linalg.norm(bs_beamforming)
        if norm > 0:
            bs_beamforming = bs_beamforming / norm

        # 转换AN波束成形向量为复数形式
        an_beamforming = np.zeros(bs_antennas, dtype=np.complex128)
        for i in range(bs_antennas):
            real_part = an_beamforming_raw[2 * i]
            imag_part = an_beamforming_raw[2 * i + 1]
            amplitude = np.sqrt(real_part ** 2 + imag_part ** 2)
            if amplitude > 0:
                an_beamforming[i] = complex(real_part / amplitude, imag_part / amplitude)
            else:
                an_beamforming[i] = complex(1, 0)

        # 标准化AN波束成形向量
        norm = np.linalg.norm(an_beamforming)
        if norm > 0:
            an_beamforming = an_beamforming / norm

        # 转换RIS相移
        ris_phase_shift = (ris_phase_raw + 1) * np.pi  # 映射到[0, 2π]

        # 转换UAV位置调整
        uav_position_delta = uav_pos_raw * 10  # 每步最大移动10米

        # 转换功率分配比例
        power_min = self.config['power_splitting_min']
        power_max = self.config['power_splitting_max']
        power_splitting_ratio = (power_splitting_raw + 1) / 2  # 映射到[0, 1]
        power_splitting_ratio = power_min + (power_max - power_min) * power_splitting_ratio  # 映射到[power_min, power_max]

        return {
            'bs_beamforming': bs_beamforming,
            'an_beamforming': an_beamforming,
            'ris_phase_shift': ris_phase_shift,
            'uav_position_delta': uav_position_delta,
            'power_splitting_ratio': power_splitting_ratio
        }

    def _build_state(self):
        """构建状态向量"""
        # 将复数信道转换为实部和虚部
        bs_u_channel_flat = np.concatenate([np.real(self.bs_u_channel).flatten(),
                                            np.imag(self.bs_u_channel).flatten()])
        u_bob_channel_flat = np.concatenate([np.real(self.u_bob_channel).flatten(),
                                             np.imag(self.u_bob_channel).flatten()])
        u_eve_channel_flat = np.concatenate([np.real(self.u_eve_channel).flatten(),
                                             np.imag(self.u_eve_channel).flatten()])

        # 组合状态向量
        state = np.concatenate([
            bs_u_channel_flat,
            u_bob_channel_flat,
            u_eve_channel_flat,
            self.uav_position,
            [self.uav_energy]
        ]).astype(np.float32)

        return state

    def _generate_bs_u_channel(self):
        """生成BS到UAV-RIS的信道"""
        bs_antennas = self.config['bs_antennas']
        ris_elements = self.config['ris_elements']
        bs_position = np.array(self.config['bs_position'])

        # 计算BS到UAV的距离
        distance = np.linalg.norm(self.uav_position - bs_position)

        # 计算路径损耗
        path_loss = self._calculate_path_loss(distance)

        # 生成莱斯衰落信道
        K = self.config['rice_factor']  # 莱斯因子

        # 生成LoS分量
        theta = np.arctan2(self.uav_position[1] - bs_position[1],
                           self.uav_position[0] - bs_position[0])
        phi = np.arcsin((self.uav_position[2] - bs_position[2]) / distance)

        H_los = np.zeros((ris_elements, bs_antennas), dtype=np.complex128)

        for m in range(ris_elements):
            for n in range(bs_antennas):
                # 这里简化处理，实际中需要考虑天线阵列结构
                phase = 2 * np.pi * (n * np.sin(theta) * np.cos(phi) + m * np.sin(phi))
                H_los[m, n] = np.exp(1j * phase)

        # 生成NLoS分量 (Rayleigh衰落)
        H_nlos_real = np.random.normal(0, 1, (ris_elements, bs_antennas))
        H_nlos_imag = np.random.normal(0, 1, (ris_elements, bs_antennas))
        H_nlos = (H_nlos_real + 1j * H_nlos_imag) / np.sqrt(2)

        # 合并LoS和NLoS分量
        H = np.sqrt(path_loss) * (np.sqrt(K / (K + 1)) * H_los + np.sqrt(1 / (K + 1)) * H_nlos)

        return H

    def _generate_u_bob_channel(self):
        """生成UAV-RIS到Bob的信道"""
        ris_elements = self.config['ris_elements']
        bob_position = np.array(self.config['bob_position'])

        # 计算UAV到Bob的距离
        distance = np.linalg.norm(self.uav_position - bob_position)

        # 计算LoS概率
        p_los = self._calculate_los_probability(distance)

        # 计算路径损耗
        path_loss = self._calculate_path_loss(distance)

        # 生成信道
        if np.random.random() < p_los:
            # LoS信道
            theta = np.arctan2(bob_position[1] - self.uav_position[1],
                               bob_position[0] - self.uav_position[0])
            phi = np.arcsin((bob_position[2] - self.uav_position[2]) / distance)

            h = np.zeros(ris_elements, dtype=np.complex128)

            for m in range(ris_elements):
                # 这里简化处理，实际中需要考虑RIS阵列结构
                phase = 2 * np.pi * m * np.sin(phi)
                h[m] = np.exp(1j * phase)

            # 添加一些随机性
            h *= np.sqrt(path_loss)
        else:
            # NLoS信道 (Rayleigh衰落)
            h_real = np.random.normal(0, 1, ris_elements)
            h_imag = np.random.normal(0, 1, ris_elements)
            h = np.sqrt(path_loss) * (h_real + 1j * h_imag) / np.sqrt(2)

        return h

    def _generate_u_eve_channel(self):
        """生成UAV-RIS到Eve的信道"""
        ris_elements = self.config['ris_elements']
        eve_position = np.array(self.config['eve_position'])

        # 计算UAV到Eve的距离
        distance = np.linalg.norm(self.uav_position - eve_position)

        # 计算LoS概率
        p_los = self._calculate_los_probability(distance)

        # 计算路径损耗
        path_loss = self._calculate_path_loss(distance)

        # 生成信道
        if np.random.random() < p_los:
            # LoS信道
            theta = np.arctan2(eve_position[1] - self.uav_position[1],
                               eve_position[0] - self.uav_position[0])
            phi = np.arcsin((eve_position[2] - self.uav_position[2]) / distance)

            h = np.zeros(ris_elements, dtype=np.complex128)

            for m in range(ris_elements):
                # 这里简化处理，实际中需要考虑RIS阵列结构
                phase = 2 * np.pi * m * np.sin(phi)
                h[m] = np.exp(1j * phase)

            # 添加一些随机性
            h *= np.sqrt(path_loss)
        else:
            # NLoS信道 (Rayleigh衰落)
            h_real = np.random.normal(0, 1, ris_elements)
            h_imag = np.random.normal(0, 1, ris_elements)
            h = np.sqrt(path_loss) * (h_real + 1j * h_imag) / np.sqrt(2)

        return h

    def _calculate_los_probability(self, distance):
        """计算LoS概率"""
        a = self.config['los_coefficient']
        b = self.config['nlos_coefficient']

        # 计算仰角 (弧度)
        elevation_angle = np.arcsin((self.uav_position[2] - 1.5) / distance)
        elevation_degree = elevation_angle * 180 / np.pi

        # LoS概率模型
        p_los = 1 / (1 + a * np.exp(-b * (elevation_degree - a)))

        return p_los

    def _calculate_path_loss(self, distance):
        """计算路径损耗"""
        # 根据路径损耗模型计算
        fc = self.config['carrier_frequency']
        alpha = self.config['path_loss_exponent']

        # 自由空间路径损耗
        lambda_c = c / fc  # 波长
        path_loss_db = 20 * np.log10(4 * np.pi * distance / lambda_c) + 10 * alpha * np.log10(distance)

        # 添加阴影衰落
        shadow_fading = np.random.normal(0, 4)  # 标准差为4dB的对数正态阴影衰落
        path_loss_db += shadow_fading

        # 转换为线性尺度
        path_loss = 10 ** (-path_loss_db / 10)

        return path_loss

    def _calculate_secrecy_rate(self, bs_beamforming, an_beamforming, ris_phase_shift, power_splitting_ratio):
        """计算保密率"""
        # 构建RIS相移矩阵
        Phi = np.diag(np.exp(1j * ris_phase_shift))

        # BS到Bob的等效信道
        h_bs_bob = np.dot(self.u_bob_channel, np.dot(Phi, self.bs_u_channel))

        # BS到Eve的等效信道
        h_bs_eve = np.dot(self.u_eve_channel, np.dot(Phi, self.bs_u_channel))

        # 计算BS发射功率
        P_tx = 10 ** (self.config['bs_max_power'] / 10) / 1000  # 转换dBm为W

        # 计算噪声功率
        noise_power = 10 ** (self.config['noise_power'] / 10) / 1000  # 转换dBm为W

        # 计算Bob的SINR (Signal to Interference plus Noise Ratio)
        # 有用信号功率
        signal_power_bob = power_splitting_ratio * P_tx * np.abs(np.dot(h_bs_bob, bs_beamforming)) ** 2

        # 人工噪声功率
        an_power_bob = power_splitting_ratio * P_tx * np.abs(np.dot(h_bs_bob, an_beamforming)) ** 2

        # Bob的SINR
        sinr_bob = signal_power_bob / (an_power_bob + noise_power)

        # 计算Eve的SINR
        # 有用信号功率
        signal_power_eve = P_tx * np.abs(np.dot(h_bs_eve, bs_beamforming)) ** 2

        # 人工噪声功率
        an_power_eve = P_tx * np.abs(np.dot(h_bs_eve, an_beamforming)) ** 2

        # Eve的SINR
        sinr_eve = signal_power_eve / (an_power_eve + noise_power)

        # 计算Bob和Eve的信道容量
        if sinr_bob > 0:
            capacity_bob = np.log2(1 + sinr_bob)
        else:
            capacity_bob = 0

        if sinr_eve > 0:
            capacity_eve = np.log2(1 + sinr_eve)
        else:
            capacity_eve = 0

        # 计算保密率
        secrecy_rate = max(0, capacity_bob - capacity_eve)

        return secrecy_rate

    def _calculate_harvested_energy(self, bs_beamforming, an_beamforming, ris_phase_shift, power_splitting_ratio):
        """计算收集的能量"""
        # 构建RIS相移矩阵
        Phi = np.diag(np.exp(1j * ris_phase_shift))

        # BS到UAV-RIS的等效信道
        h_bs_ris = self.bs_u_channel

        # 计算BS发射功率
        P_tx = 10 ** (self.config['bs_max_power'] / 10) / 1000  # 转换dBm为W

        # 收集能量的比例
        energy_proportion = 1 - power_splitting_ratio

        # 有用信号功率
        signal_power = energy_proportion * P_tx * np.sum(np.abs(np.dot(h_bs_ris, bs_beamforming)) ** 2)

        # 人工噪声功率
        an_power = energy_proportion * P_tx * np.sum(np.abs(np.dot(h_bs_ris, an_beamforming)) ** 2)

        # 总接收功率
        total_power = signal_power + an_power

        # 能量转换效率
        eta = self.config['energy_harvesting_efficiency']

        # 收集的能量
        harvested_energy = eta * total_power

        return harvested_energy

    def _calculate_energy_efficiency(self, secrecy_rate, harvested_energy):
        """计算能量效率"""
        # 计算BS发射功率
        P_tx = 10 ** (self.config['bs_max_power'] / 10) / 1000  # 转换dBm为W

        # UAV的功耗模型
        P_uav = 0.1  # 简化的UAV功耗 (W)

        # 总功耗
        total_power = P_tx + P_uav - harvested_energy

        # 避免除以零
        if total_power <= 0:
            return 0

        # 能量效率: 保密率/总功耗
        energy_efficiency = secrecy_rate / total_power

        return energy_efficiency

    def plot_comparison(self, ris_elements_range, antenna_range, filename="ris_antenna_impact.png"):
        """绘制RIS元件数量和基站天线数量的影响对比图"""
        # 创建图形
        plt.figure(figsize=(12, 8))

        # 设置颜色和标记
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        markers = ['o', 's', '^', 'v', 'D', '*']

        # 1. 绘制RIS元件数量对保密率的影响
        secrecy_rates_ris = []
        for ris_elements in ris_elements_range:
            # 简化的分析模型：随着RIS元件增加，保密率提高
            # 在实际中，应该使用真实的计算模型或经验数据
            secrecy_rate = 0.5 + 0.05 * np.log2(ris_elements / 10)
            secrecy_rates_ris.append(secrecy_rate)

        plt.subplot(2, 1, 1)
        plt.plot(ris_elements_range, secrecy_rates_ris, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of RIS Elements', fontsize=12)
        plt.ylabel('Secrecy Rate (bps/Hz)', fontsize=12)
        plt.title('Impact of RIS Elements on Secrecy Rate', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 2. 绘制基站天线数量对保密率的影响
        secrecy_rates_antenna = []
        for antennas in antenna_range:
            # 简化的分析模型：随着天线数量增加，保密率提高
            secrecy_rate = 0.3 + 0.1 * np.log2(antennas)
            secrecy_rates_antenna.append(secrecy_rate)

        plt.subplot(2, 1, 2)
        plt.plot(antenna_range, secrecy_rates_antenna, 'r-s', linewidth=2, markersize=8)
        plt.xlabel('Number of BS Antennas', fontsize=12)
        plt.ylabel('Secrecy Rate (bps/Hz)', fontsize=12)
        plt.title('Impact of BS Antennas on Secrecy Rate', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.config['store_path'], filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"比较图已保存至: {save_path}")

        # 保存数据
        comparison_data = {
            'ris_elements': ris_elements_range,
            'secrecy_rates_ris': secrecy_rates_ris,
            'antenna_range': antenna_range,
            'secrecy_rates_antenna': secrecy_rates_antenna
        }
        data_path = os.path.join(self.config['store_path'], 'comparison_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(comparison_data, f)

    def plot_power_allocation(self, filename="power_allocation.png"):
        """绘制功率分配比例的变化"""
        if len(self.power_allocation_data) == 0:
            print("没有足够的功率分配数据进行绘图")
            return

        plt.figure(figsize=(10, 6))

        # 绘制功率分配比例随时间变化曲线
        steps = np.arange(1, len(self.power_allocation_data) + 1)
        plt.plot(steps, self.power_allocation_data, 'g-', linewidth=2)

        # 绘制最佳功率分配比例 (如果有)
        if self.best_config and 'power_splitting_ratio' in self.best_config:
            best_ratio = self.best_config['power_splitting_ratio']
            plt.axhline(y=best_ratio, color='r', linestyle='--', linewidth=2,
                        label=f'Best Ratio: {best_ratio:.3f}')
            plt.legend(fontsize=12)

        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Power Splitting Ratio (ρ)', fontsize=12)
        plt.title('Power Allocation Ratio Evolution', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加说明
        info_text = (
            f"ρ: Information Decoding Portion\n"
            f"1-ρ: Energy Harvesting Portion\n"
            f"Min ρ: {self.config['power_splitting_min']}\n"
            f"Max ρ: {self.config['power_splitting_max']}"
        )
        plt.annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
                     fontsize=10)

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.config['store_path'], filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"功率分配图已保存至: {save_path}")

        # 保存数据
        power_data = {
            'steps': steps.tolist(),
            'power_allocation': self.power_allocation_data,
            'best_ratio': self.best_config['power_splitting_ratio'] if self.best_config else None
        }
        data_path = os.path.join(self.config['store_path'], 'power_allocation_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(power_data, f)

    def plot_transmit_power(self, filename="transmit_power.png"):
        """绘制发射功率与保密率的关系"""
        if len(self.secrecy_rates) == 0:
            print("没有足够的保密率数据进行绘图")
            return

        plt.figure(figsize=(10, 6))

        # 假设使用了不同的发射功率级别进行测试
        # 这里我们使用一个理论模型来绘制曲线

        # 理论功率范围 (dBm)
        power_range = np.arange(10, 41, 1)

        # 简化模型：保密率随发射功率对数增长
        theoretical_secrecy_rates = [0.1 * np.log2(1 + p / 10) for p in power_range]

        # 绘制理论曲线
        plt.plot(power_range, theoretical_secrecy_rates, 'b-', linewidth=2, label='Theoretical Model')

        # 添加当前系统的工作点
        current_power = self.config['bs_max_power']
        avg_secrecy_rate = np.mean(self.secrecy_rates[-10:]) if len(self.secrecy_rates) >= 10 else np.mean(
            self.secrecy_rates)

        plt.scatter(current_power, avg_secrecy_rate, color='r', s=100, marker='*',
                    label=f'Current: {current_power} dBm, {avg_secrecy_rate:.3f} bps/Hz')

        # 添加标签和图例
        plt.xlabel('Transmit Power (dBm)', fontsize=12)
        plt.ylabel('Secrecy Rate (bps/Hz)', fontsize=12)
        plt.title('Relationship Between Transmit Power and Secrecy Rate', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # 添加额外信息
        if self.best_secrecy_rate > 0:
            plt.axhline(y=self.best_secrecy_rate, color='g', linestyle='--', linewidth=1,
                        label=f'Best Achieved: {self.best_secrecy_rate:.3f} bps/Hz')

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.config['store_path'], filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"发射功率图已保存至: {save_path}")

        # 保存数据
        power_data = {
            'power_range': power_range.tolist(),
            'theoretical_rates': theoretical_secrecy_rates,
            'current_power': current_power,
            'avg_secrecy_rate': avg_secrecy_rate,
            'best_secrecy_rate': self.best_secrecy_rate
        }
        data_path = os.path.join(self.config['store_path'], 'transmit_power_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(power_data, f)

    def save_results(self):
        """保存训练结果到文件"""
        # 确保目录存在
        os.makedirs(self.config['store_path'], exist_ok=True)

        # 收集结果数据
        results = {
            'config': self.config,
            'secrecy_rates': self.secrecy_rates,
            'energy_efficiency': self.energy_efficiency,
            'best_secrecy_rate': self.best_secrecy_rate,
            'best_energy_harvested': self.best_energy_harvested,
            'best_energy_efficiency': self.best_energy_efficiency,
            'best_config': self.best_config
        }

        # 保存到文件
        save_path = os.path.join(self.config['store_path'], 'training_results.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"训练结果已保存至: {save_path}")

        # 生成结果摘要
        summary = (
            f"=== UAV-RIS SWIPT System Results Summary ===\n"
            f"RIS Elements: {self.config['ris_elements']}\n"
            f"BS Antennas: {self.config['bs_antennas']}\n"
            f"BS Power: {self.config['bs_max_power']} dBm\n\n"
            f"Best Secrecy Rate: {self.best_secrecy_rate:.4f} bps/Hz\n"
            f"Best Energy Harvested: {self.best_energy_harvested:.4f} W\n"
            f"Best Energy Efficiency: {self.best_energy_efficiency:.4f} bps/Hz/W\n"
        )

        if self.best_config:
            summary += (
                f"\nBest UAV Position: [{self.best_config['uav_position'][0]:.2f}, "
                f"{self.best_config['uav_position'][1]:.2f}, {self.best_config['uav_position'][2]:.2f}] m\n"
                f"Best Power Splitting Ratio: {self.best_config['power_splitting_ratio']:.4f}\n"
            )

        # 保存摘要到文本文件
        summary_path = os.path.join(self.config['store_path'], 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)

        print(f"结果摘要已保存至: {summary_path}")