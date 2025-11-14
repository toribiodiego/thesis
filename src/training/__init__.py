"""Training utilities for DQN."""

from .dqn_trainer import hard_update_target, init_target_network

__all__ = ['hard_update_target', 'init_target_network']
