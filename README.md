# DDPG-MuJoCo

Deep reinforcement learning implementation for MuJoCo environments, specifically trained on Humanoid-v5.

## Project Evolution

This project started with a Deep Deterministic Policy Gradient (DDPG) implementation. However, stability issues were encountered during training - the agent struggled to learn consistently and performance was unreliable.

To address these challenges, the implementation was transitioned to Twin Delayed Deep Deterministic Policy Gradient (TD3), which incorporates several improvements over DDPG:
- Twin Q-networks to reduce overestimation bias
- Delayed policy updates for more stable learning
- Target policy smoothing to reduce variance

The TD3 implementation proved much more stable and achieved better training performance.

## Running

Training: `python train.py`

Visualization: `python visualize.py`
