# 修订版配置文件，适用于UAV-RIS一体化系统
CONFIG = {
    # 一般参数
    'device': 'cuda',  # 使用GPU，如果没有则会自动切换到CPU
    'seed': 42,
    'save_dir': 'results/maritime_td3',

    # 环境参数
    'num_ue': 1,  # 合法用户数量
    'num_eve': 1,  # 窃听者数量
    'ris_elements': 16,  # RIS反射元件数量 (4x4 UPA)
    'bs_antennas': 4,  # 基站天线数量
    'time_slots': 100,  # 仿真总时隙数
    'time_step': 0.1,  # 每个时隙的时间长度(s)
    'enable_eh': True,  # 开启能量收集
    'frequency': 28e9,  # 载波频率(28GHz)
    'boundary': [
        (-100, 100),  # x范围
        (-100, 100),  # y范围
        (0, 100)  # z范围
    ],

    # 训练参数
    'num_episodes': 1000,  # 训练的总回合数
    'batch_size': 64,  # 较小的批量大小加速初期训练
    'buffer_size': 100000,  # 适中的缓冲区大小
    'hidden_dim': 256,
    'actor_lr': 1e-3,  # 更高的学习率
    'critic_lr': 1e-3,  # 更高的学习率
    'discount': 0.99,
    'tau': 0.01,  # 更快的目标网络更新
    'policy_noise': 0.5,  # 更大的策略噪声
    'noise_clip': 0.5,
    'policy_freq': 2,
    'exploration_noise': 0.5,  # 更大的探索噪声

    # 评估参数
    'eval_freq': 10,  # 评估频率（回合数）
    'eval_episodes': 5,  # 每次评估的回合数
    'render': True,  # 是否渲染环境

    # 初始位置
    'init_positions': {
        'bs': [0, 0, 30],  # 基站位置
        'uav-ris': [0, 25, 50],  # UAV-RIS初始位置
        'users': [
            [4, 47, 0],  # 用户1初始位置
            [25, 25, 0]  # 用户2初始位置
        ],
        'eves': [
            [47, -4, 0]  # 窃听者初始位置
        ]
    },

    # 能量参数
    'battery_capacity': 100,  # UAV-RIS电池容量(Wh)
    'energy_conversion_efficiency': 0.7,  # 能量转换效率
    'min_energy_threshold': 0.2,  # 最低电量阈值(占总容量比例)

    # 通信参数
    'bs_max_power': 30,  # 基站最大发射功率(dBm)
    'noise_power': -114,  # 噪声功率(dBm)
    'min_data_rate': 0.5,  # 最低数据速率要求(bps/Hz)
}