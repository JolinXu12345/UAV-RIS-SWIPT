This project implements a simulation of a maritime secure communication system assisted by a UAV (Unmanned Aerial Vehicle) and an RIS (Reconfigurable Intelligent Surface) based on the Twin Delayed DDPG (TD3) algorithm. The system leverages reinforcement learning to optimize UAV positioning, BS beamforming, and RIS phase shifts in order to maximize the secrecy rate and energy efficiency.

## Project Structure

- `channel.py`: Implements the channel models, including mmWave channels and maritime channels
- `entity.py`: Defines system entities (UAV, RIS, users, etc.)
- `environment.py`: Implements the maritime communication environment and the reinforcement-learning environment
- `td3.py`: Implements the TD3 algorithm
- `run_simulation.py`: Simulation execution script
- `main.py`: Simplified main program for quick testing
- `config_example.py`: Example configuration file

## System Model

The system comprises the following key components:

1. **Base Station (BS)**: A fixed multi-antenna base station serving as the information source
2. **Unmanned Aerial Vehicle (UAV)**: A mobile platform equipped with an RIS to assist communication
3. **Legitimate User (UE)**: The intended recipient of the transmitted information
4. **Eavesdropper (Eve)**: A malicious user attempting to intercept the information
5. **Channel Models**: Incorporate large-scale path loss and small-scale fast fading, taking maritime environmental characteristics into account

The system also incorporates an energy-harvesting feature to extend the UAV’s operational endurance.

## System Requirements

- Python 3.8+
- PyTorch 1.13.1+
- NumPy 1.24.3+
- Matplotlib 3.7.1+
- SciPy 1.10.1+