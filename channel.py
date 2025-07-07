import numpy as np
import math
import cmath


class Channel:
    """
    基础信道类，实现基本的信道计算功能
    """

    def __init__(self, transmitter, receiver, frequency, name=''):
        """
        初始化信道

        参数:
            transmitter: 发射端对象
            receiver: 接收端对象
            frequency: 载波频率(Hz)
            name: 信道名称
        """
        self.transmitter = transmitter
        self.receiver = receiver
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self.channel_name = name

        # 初始化路径损耗参数
        self.path_loss_exponent = 2.0  # 默认路径损耗指数
        self.shadow_fading_std = 3.0  # 默认阴影衰落标准差(dB)

        # 初始化小尺度衰落参数
        self.rician_factor_dB = 3.0  # 莱斯因子(dB)

        # 计算信道矩阵
        self.channel_matrix = None
        self.update_channel()

    def get_distance(self):
        """计算发射端与接收端之间的距离"""
        return np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate)

    def dB_to_linear(self, dB_value):
        """将dB值转换为线性值"""
        return 10 ** (dB_value / 10)

    def linear_to_dB(self, linear_value):
        """将线性值转换为dB值"""
        return 10 * np.log10(linear_value)

    def get_path_loss(self):
        """
        计算路径损耗(线性形式)

        返回:
            路径损耗(线性)
        """
        distance = self.get_distance()

        # 自由空间路径损耗(dB)
        free_space_loss_dB = 20 * math.log10(4 * math.pi * distance / self.wavelength)

        # 路径损耗指数引起的额外损耗(dB)
        additional_loss_dB = 10 * (self.path_loss_exponent - 2) * math.log10(distance)

        # 阴影衰落(dB)
        shadow_fading_dB = np.random.normal(0, self.shadow_fading_std)

        # 总路径损耗(dB)
        total_loss_dB = free_space_loss_dB + additional_loss_dB + shadow_fading_dB

        # 返回线性形式的路径损耗(注意是损耗的倒数，即增益)
        return self.dB_to_linear(-total_loss_dB)

    def calculate_channel_matrix(self):
        """
        计算信道矩阵，由子类实现
        """
        pass

    def update_channel(self):
        """
        更新信道状态
        """
        self.channel_matrix = self.calculate_channel_matrix()


class BSToUAVChannel(Channel):
    """
    基站到UAV-RIS的信道模型
    考虑大尺度和小尺度衰落的复合信道
    """

    def __init__(self, base_station, uav_ris, frequency):
        """
        初始化BS到UAV-RIS的信道

        参数:
            base_station: 基站对象
            uav_ris: UAV-RIS一体化对象
            frequency: 载波频率(Hz)
        """
        super().__init__(base_station, uav_ris, frequency, name='H_BS_UAV')

        # 设置BS-UAV信道特定参数
        self.path_loss_exponent = 2.2  # BS到UAV的路径损耗指数
        self.shadow_fading_std = 3.0  # BS到UAV的阴影衰落(dB)
        self.rician_factor_dB = 3.0  # BS到UAV的莱斯因子(dB)

    def calculate_channel_matrix(self):
        """
        计算BS到UAV-RIS的信道矩阵

        返回:
            信道矩阵，形状为(Nr, Nt)，其中Nr是UAV-RIS的天线/反射元件数量，Nt是BS的天线数量
        """
        # 获取发射端和接收端的天线数量
        Nt = self.transmitter.ant_num  # BS天线数量
        Nr = self.receiver.ris_elements  # UAV-RIS反射元件数量

        # 初始化信道矩阵
        H = np.zeros((Nr, Nt), dtype=complex)

        # 计算路径损耗
        path_loss = math.sqrt(self.get_path_loss())

        # 计算莱斯因子(线性)
        K = self.dB_to_linear(self.rician_factor_dB)

        # 计算LoS分量和NLoS分量的权重
        los_weight = math.sqrt(K / (K + 1))
        nlos_weight = math.sqrt(1 / (K + 1))

        # 计算BS到UAV-RIS的距离
        distance = self.get_distance()

        # 计算相位偏移
        phase_shift = 2 * math.pi * distance / self.wavelength

        # 计算BS和UAV-RIS之间的方向矢量
        direction = (self.receiver.coordinate - self.transmitter.coordinate) / distance

        # 计算LoS分量
        for i in range(Nr):
            for j in range(Nt):
                # 计算天线位置引起的相位差
                # 这里使用简化计算，实际应考虑阵列几何和方向
                phase_diff = 0.5 * math.pi * (i * j / (Nr * Nt))  # 简化的相位差模型

                # LoS分量
                los_component = los_weight * np.exp(1j * (phase_shift + phase_diff))

                # NLoS分量(复高斯随机变量)
                nlos_real = np.random.normal(0, 1 / math.sqrt(2))
                nlos_imag = np.random.normal(0, 1 / math.sqrt(2))
                nlos_component = nlos_weight * complex(nlos_real, nlos_imag)

                # 组合LoS和NLoS分量
                H[i, j] = path_loss * (los_component + nlos_component)

        return H


class UAVToUEChannel(Channel):
    """
    UAV-RIS到用户的信道模型
    考虑LoS和NLoS的概率和海事环境
    """

    def __init__(self, uav_ris, user, frequency, is_eavesdropper=False):
        """
        初始化UAV-RIS到用户的信道

        参数:
            uav_ris: UAV-RIS一体化对象
            user: 用户对象
            frequency: 载波频率(Hz)
            is_eavesdropper: 是否为窃听者
        """
        # 先初始化这些属性，然后再调用父类构造函数
        self.los_param_a = 5.0  # 环境相关参数a
        self.los_param_b = 0.5  # 环境相关参数b
        self.is_eavesdropper = is_eavesdropper

        name = f'h_UAV_Eve{user.index}' if is_eavesdropper else f'h_UAV_UE{user.index}'
        super().__init__(uav_ris, user, frequency, name=name)

        # 设置UAV-UE信道特定参数
        self.path_loss_exponent = 2.8  # UAV-RIS到用户的路径损耗指数
        self.shadow_fading_std = 3.0  # UAV-RIS到用户的阴影衰落(dB)

        # 海事环境特定参数
        self.los_param_a = 5.0  # 环境相关参数a
        self.los_param_b = 0.5  # 环境相关参数b
        self.is_eavesdropper = is_eavesdropper

    def get_los_probability(self):
        """
        计算UAV-RIS到用户的LoS概率

        返回:
            LoS概率 [0,1]
        """
        # 计算距离
        distance = self.get_distance()

        # 计算仰角(弧度)
        height_diff = abs(self.transmitter.coordinate[2] - self.receiver.coordinate[2])
        elevation_angle = math.atan2(height_diff, distance)

        # 根据仰角计算LoS概率
        p_los = 1 / (1 + self.los_param_a * math.exp(-self.los_param_b * (elevation_angle - self.los_param_a)))

        return p_los

    def calculate_channel_matrix(self):
        """
        计算UAV-RIS到用户的信道矩阵

        返回:
            信道矩阵，形状为(Nu, Nr)，其中Nu是用户的天线数量，Nr是UAV-RIS的反射元件数量
        """
        # 获取发射端和接收端的天线数量
        Nr = self.transmitter.ris_elements  # UAV-RIS反射元件数量
        Nu = self.receiver.ant_num  # 用户天线数量(通常为1)

        # 初始化信道矩阵
        H = np.zeros((Nu, Nr), dtype=complex)

        # 计算LoS概率
        p_los = self.get_los_probability()

        # 决定当前是LoS还是NLoS
        is_los = np.random.random() < p_los

        # 计算路径损耗
        path_loss = math.sqrt(self.get_path_loss())

        # 计算距离
        distance = self.get_distance()

        # 计算相位偏移
        phase_shift = 2 * math.pi * distance / self.wavelength

        # 计算UAV-RIS和用户之间的方向矢量
        direction = (self.receiver.coordinate - self.transmitter.coordinate) / distance

        # 根据LoS状态计算信道矩阵
        if is_los:
            # LoS状态下的信道模型
            K = self.dB_to_linear(self.rician_factor_dB)  # 莱斯因子
            los_weight = math.sqrt(K / (K + 1))
            nlos_weight = math.sqrt(1 / (K + 1))

            for i in range(Nu):
                for j in range(Nr):
                    # 计算天线位置引起的相位差
                    phase_diff = 0.5 * math.pi * (i * j / (Nu * Nr))  # 简化的相位差模型

                    # LoS分量
                    los_component = los_weight * np.exp(1j * (phase_shift + phase_diff))

                    # NLoS分量(复高斯随机变量)
                    nlos_real = np.random.normal(0, 1 / math.sqrt(2))
                    nlos_imag = np.random.normal(0, 1 / math.sqrt(2))
                    nlos_component = nlos_weight * complex(nlos_real, nlos_imag)

                    # 组合LoS和NLoS分量
                    H[i, j] = path_loss * (los_component + nlos_component)
        else:
            # NLoS状态下的信道模型(瑞利衰落)
            for i in range(Nu):
                for j in range(Nr):
                    # 生成复高斯随机变量
                    real_part = np.random.normal(0, 1 / math.sqrt(2))
                    imag_part = np.random.normal(0, 1 / math.sqrt(2))

                    # NLoS分量
                    H[i, j] = path_loss * complex(real_part, imag_part)

        return H


class MaritimeChannel:
    """
    海事环境下的信道管理类
    管理所有通信链路的信道
    """

    def __init__(self, base_station, uav_ris, users, eavesdroppers, frequency):
        """
        初始化海事信道管理器

        参数:
            base_station: 基站对象
            uav_ris: UAV-RIS一体化对象
            users: 合法用户列表
            eavesdroppers: 窃听者列表
            frequency: 载波频率(Hz)
        """
        self.bs = base_station
        self.uav_ris = uav_ris
        self.users = users
        self.eavesdroppers = eavesdroppers
        self.frequency = frequency

        # 初始化BS到UAV-RIS的信道
        self.h_bs_uav = BSToUAVChannel(base_station, uav_ris, frequency)

        # 初始化UAV-RIS到合法用户的信道
        self.h_uav_ue = []
        for user in users:
            self.h_uav_ue.append(UAVToUEChannel(uav_ris, user, frequency, is_eavesdropper=False))

        # 初始化UAV-RIS到窃听者的信道
        self.h_uav_eve = []
        for eve in eavesdroppers:
            self.h_uav_eve.append(UAVToUEChannel(uav_ris, eve, frequency, is_eavesdropper=True))

    def update_channels(self):
        """
        更新所有信道
        """
        # 更新BS到UAV-RIS的信道
        self.h_bs_uav.update_channel()

        # 更新UAV-RIS到合法用户的信道
        for h in self.h_uav_ue:
            h.update_channel()

        # 更新UAV-RIS到窃听者的信道
        for h in self.h_uav_eve:
            h.update_channel()

    def get_effective_channel(self, user_index, is_eavesdropper=False):
        """
        计算有效端到端信道

        参数:
            user_index: 用户索引
            is_eavesdropper: 是否为窃听者

        返回:
            有效信道矩阵
        """
        # 获取UAV-RIS相移矩阵
        theta = self.uav_ris.get_phase_shift_matrix()

        # 获取BS到UAV-RIS的信道
        h_bs_uav = self.h_bs_uav.channel_matrix

        # 获取UAV-RIS到用户的信道
        if is_eavesdropper:
            h_uav_user = self.h_uav_eve[user_index].channel_matrix
        else:
            h_uav_user = self.h_uav_ue[user_index].channel_matrix

        # 计算级联信道
        effective_channel = h_uav_user @ theta @ h_bs_uav

        return effective_channel