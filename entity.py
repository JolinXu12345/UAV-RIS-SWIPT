import numpy as np
import math


class BaseStation:
    """
    基站实体类
    """

    def __init__(self, coordinate=[0, 0, 30], ant_num=4, max_power=30):
        """
        初始化基站

        参数:
            coordinate: 基站坐标 [x, y, z]
            ant_num: 天线数量
            max_power: 最大发射功率(dBm)
        """
        self.type = 'BS'
        self.coordinate = np.array(coordinate, dtype=float)
        self.ant_num = ant_num
        self.max_power = max_power  # dBm

        # 波束成形矩阵 (天线数 x 用户数)
        self.beamforming_matrix = None

        # 人工噪声矩阵
        self.an_matrix = None

        # 功率分配
        self.info_power_ratio = 0.8  # 信息功率比例
        self.an_power_ratio = 0.2  # 人工噪声功率比例

    def set_beamforming(self, beamforming_matrix):
        """
        设置波束成形矩阵

        参数:
            beamforming_matrix: 波束成形矩阵，形状为(天线数, 用户数)
        """
        self.beamforming_matrix = beamforming_matrix

    def set_an_matrix(self, an_matrix):
        """
        设置人工噪声矩阵

        参数:
            an_matrix: 人工噪声矩阵
        """
        self.an_matrix = an_matrix

    def set_power_allocation(self, info_power_ratio):
        """
        设置功率分配比例

        参数:
            info_power_ratio: 分配给信息传输的功率比例 [0,1]
        """
        self.info_power_ratio = info_power_ratio
        self.an_power_ratio = 1 - info_power_ratio

    def get_transmit_power(self):
        """
        获取发射功率(线性值)

        返回:
            发射功率(W)
        """
        return 10 ** (self.max_power / 10) / 1000  # dBm转W


class UAVRIS:
    """
    UAV-RIS一体化实体类
    """

    def __init__(self, coordinate=[0, 25, 50], ris_elements=16, ris_rows=4, ris_cols=4,
                 max_movement=0.5, height=50, battery_capacity=100, energy_conversion_efficiency=0.7):
        """
        初始化UAV-RIS一体化设备

        参数:
            coordinate: UAV-RIS坐标 [x, y, z]
            ris_elements: RIS反射元件总数
            ris_rows: RIS行数
            ris_cols: RIS列数
            max_movement: 每时隙最大移动距离(m)
            height: 固定飞行高度(m)
            battery_capacity: 电池容量(Wh)
            energy_conversion_efficiency: 能量转换效率 [0,1]
        """
        self.type = 'UAV-RIS'
        self.coordinate = np.array(coordinate, dtype=float)
        self.ris_elements = ris_elements
        self.ris_rows = ris_rows
        self.ris_cols = ris_cols
        self.max_movement = max_movement
        self.fixed_height = height

        # 能量相关参数
        self.battery_capacity = battery_capacity  # Wh
        self.current_energy = battery_capacity  # Wh
        self.energy_harvested = 0  # Wh (累计)
        self.energy_conversion_efficiency = energy_conversion_efficiency

        # 功率分配
        self.it_power_ratio = 0.8  # 信息传输功率比例
        self.eh_power_ratio = 0.2  # 能量收集功率比例

        # RIS相移矩阵
        self.phase_shifts = np.zeros(ris_elements)  # 相移角度 [0, 2π)

        # 运动相关
        self.velocity = np.zeros(3)  # 速度向量
        self.previous_coordinate = np.array(coordinate, dtype=float)  # 上一时隙位置

    def set_phase_shifts(self, phase_shifts):
        """
        设置RIS相移

        参数:
            phase_shifts: 相移角度数组，长度为ris_elements
        """
        if len(phase_shifts) != self.ris_elements:
            raise ValueError(f"相移数组长度({len(phase_shifts)})与反射元件数量({self.ris_elements})不匹配")

        # 确保相移在 [0, 2π) 范围内
        self.phase_shifts = np.mod(phase_shifts, 2 * np.pi)

    def get_phase_shift_matrix(self):
        """
        获取RIS相移矩阵(对角矩阵)

        返回:
            相移对角矩阵
        """
        # 将相移转换为复数形式的对角矩阵
        return np.diag(np.exp(1j * self.phase_shifts))

    def move(self, direction_vector):
        """
        移动UAV-RIS

        参数:
            direction_vector: 移动方向向量 [dx, dy, dz]

        返回:
            实际移动距离
        """
        # 保存当前位置作为上一位置
        self.previous_coordinate = self.coordinate.copy()

        # 计算移动方向和距离
        direction = np.array(direction_vector, dtype=float)
        distance = np.linalg.norm(direction)

        if distance > 0:
            # 规范化方向向量
            normalized_direction = direction / distance

            # 限制移动距离
            actual_distance = min(distance, self.max_movement)

            # 计算新位置
            new_coordinate = self.coordinate + normalized_direction * actual_distance

            # 如果有固定高度，保持高度不变
            if self.fixed_height is not None:
                new_coordinate[2] = self.fixed_height

            self.coordinate = new_coordinate

            # 更新速度
            self.velocity = (self.coordinate - self.previous_coordinate)

            return actual_distance
        else:
            self.velocity = np.zeros(3)
            return 0

    def calculate_energy_consumption(self, delta_time=0.1):
        """
        计算移动消耗的能量

        参数:
            delta_time: 时间间隔(s)

        返回:
            消耗的能量(Wh)
        """
        # 旋翼无人机能耗模型参数
        P_i = 790.6715  # 悬停诱导功率(W)
        P_0 = 580.65  # 叶片廓形功率(W)
        U_tip = 200  # 叶片尖速度(m/s)
        d_0 = 0.3  # 机身阻力比
        rho = 1.225  # 空气密度(kg/m^3)
        s = 0.05  # 旋翼实度
        A = 0.79  # 旋翼盘面积(m^2)

        # 计算悬停参考速度
        g = 9.81  # 重力加速度(m/s^2)
        m = 1.3  # 无人机质量(kg)
        T = m * g  # 悬停推力(N)
        v_0 = (T / (2 * rho * A)) ** 0.5  # 悬停引导速度(m/s)

        # 计算速度大小
        velocity = np.linalg.norm(self.velocity) / delta_time  # m/s

        # 计算功率(W)
        power = P_0 * (1 + 3 * velocity ** 2 / U_tip ** 2) + \
                0.5 * d_0 * rho * s * A * velocity ** 3 + \
                P_i * ((1 + velocity ** 4 / (4 * v_0 ** 4)) ** 0.5 - velocity ** 2 / (2 * v_0 ** 2)) ** 0.5

        # 计算能耗(Wh)
        energy = power * delta_time / 3600

        return energy

    def collect_energy(self, received_signal_power, delta_time=0.1):
        """
        收集RF能量

        参数:
            received_signal_power: 接收的信号功率(W)
            delta_time: 时间间隔(s)

        返回:
            收集的能量(Wh)
        """
        # 计算能量收集功率
        harvested_power = self.energy_conversion_efficiency * self.eh_power_ratio * received_signal_power

        # 计算收集的能量(Wh)
        harvested_energy = harvested_power * delta_time / 3600

        # 更新能量状态
        self.energy_harvested += harvested_energy
        self.current_energy += harvested_energy

        # 限制电池容量
        self.current_energy = min(self.current_energy, self.battery_capacity)

        return harvested_energy

    def set_power_splitting_ratio(self, it_ratio):
        """
        设置功率分配比例

        参数:
            it_ratio: 信息传输功率比例 [0,1]
        """
        self.it_power_ratio = it_ratio
        self.eh_power_ratio = 1 - it_ratio


class User:
    """
    用户实体类（包括合法用户和窃听者）
    """

    def __init__(self, coordinate, index=0, is_eavesdropper=False, ant_num=1, velocity=None):
        """
        初始化用户

        参数:
            coordinate: 用户坐标 [x, y, z]
            index: 用户索引
            is_eavesdropper: 是否为窃听者
            ant_num: 天线数量
            velocity: 移动速度向量 [vx, vy, vz]
        """
        self.type = 'Eve' if is_eavesdropper else 'UE'
        self.coordinate = np.array(coordinate, dtype=float)
        self.index = index
        self.is_eavesdropper = is_eavesdropper
        self.ant_num = ant_num
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity)

        # 通信参数
        self.noise_power = -114  # dBm
        self.capacity = 0  # 数据速率(bps/Hz)
        self.secrecy_rate = 0  # 保密速率(bps/Hz)
        self.sinr = 0  # 信噪比

        # QoS要求
        self.min_data_rate = 0.5  # 最低数据速率要求(bps/Hz)

    def move(self, delta_time=0.1):
        """
        根据速度移动用户

        参数:
            delta_time: 时间间隔(s)
        """
        if np.linalg.norm(self.velocity) > 0:
            self.coordinate += self.velocity * delta_time

    def get_noise_power_linear(self):
        """
        获取噪声功率（线性值）

        返回:
            噪声功率(W)
        """
        return 10 ** (self.noise_power / 10) / 1000  # dBm转W