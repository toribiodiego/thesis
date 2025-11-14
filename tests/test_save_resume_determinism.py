"""
Smoke test for save/resume determinism.

Tests that:
1. Training for N steps produces deterministic results
2. Saving a checkpoint and resuming produces identical behavior
3. Epsilon, rewards, and actions match between runs
4. Provides detailed comparison report with match/mismatch counts

This is an end-to-end integration test that verifies the complete
checkpoint/resume pipeline works correctly with deterministic execution.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.models.dqn import DQN
from src.replay.replay_buffer import ReplayBuffer
from src.training import (
    CheckpointManager,
    EpsilonScheduler,
    configure_optimizer,
    get_rng_states,
    set_rng_states,
    resume_from_checkpoint,
)
from src.training.training_loop import select_epsilon_greedy_action
from src.utils import set_seed, configure_determinism


class SimpleMockEnv:
    """Simple mock environment for deterministic testing."""

    def __init__(self, obs_shape=(4, 84, 84), num_actions=6, seed=42):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.action_space = type('ActionSpace', (), {'n': num_actions})()
        self.observation_space = type('ObservationSpace', (), {
            'shape': obs_shape
        })()
        self.seed_value = seed
        self.step_count = 0
        set_seed(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed_value = seed
            set_seed(seed)
        self.step_count = 0
        obs = np.random.randint(0, 255, self.obs_shape, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        # Deterministic transitions based on current state
        obs = np.random.randint(0, 255, self.obs_shape, dtype=np.uint8)
        reward = np.random.randn()  # Random reward
        done = self.step_count % 100 == 0  # Episode every 100 steps
        truncated = False
        info = {}
        return obs, reward, done, truncated, info


def run_training_steps(
    env,
    online_model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    epsilon_scheduler: EpsilonScheduler,
    num_steps: int,
    start_step: int = 0,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run training for a fixed number of steps and record metrics.

    Returns:
        Dictionary with:
            - epsilons: List of epsilon values
            - rewards: List of rewards
            - actions: List of actions
            - final_step: Final step count
            - checksum: Checksum of all metrics
    """
    epsilons = []
    rewards = []
    actions = []

    obs, _ = env.reset()
    state = torch.from_numpy(obs).float().unsqueeze(0).to(device)

    for step in range(start_step, start_step + num_steps):
        # Get epsilon
        epsilon = epsilon_scheduler.get_epsilon(step)
        epsilons.append(epsilon)

        # Select action
        with torch.no_grad():
            action = select_epsilon_greedy_action(
                online_model,
                state.squeeze(0),
                epsilon,
                env.num_actions
            )
        actions.append(action)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)

        # Add to replay buffer
        replay_buffer.append(
            obs, action, reward,
            next_obs, done or truncated
        )

        # Update state
        obs = next_obs
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        # Reset if done
        if done or truncated:
            obs, _ = env.reset()
            state = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        # Perform training update if buffer has enough samples
        if replay_buffer.can_sample(32) and step % 4 == 0:
            batch = replay_buffer.sample(32)
            # Move batch to device
            batch_device = {
                'states': batch['states'].to(device),
                'actions': batch['actions'].to(device),
                'rewards': batch['rewards'].to(device),
                'next_states': batch['next_states'].to(device),
                'dones': batch['dones'].to(device)
            }

            from src.training.metrics import perform_update_step
            perform_update_step(
                online_model,
                target_model,
                optimizer,
                batch_device,
                gamma=0.99,
                update_count=step
            )

    # Compute checksum
    checksum = compute_checksum(epsilons, rewards, actions)

    return {
        'epsilons': epsilons,
        'rewards': rewards,
        'actions': actions,
        'final_step': start_step + num_steps,
        'checksum': checksum
    }


def compute_checksum(epsilons: List[float], rewards: List[float], actions: List[int]) -> str:
    """Compute checksum of metrics for comparison."""
    # Convert to bytes
    eps_bytes = np.array(epsilons, dtype=np.float32).tobytes()
    rew_bytes = np.array(rewards, dtype=np.float32).tobytes()
    act_bytes = np.array(actions, dtype=np.int32).tobytes()

    # Compute hash
    hasher = hashlib.sha256()
    hasher.update(eps_bytes)
    hasher.update(rew_bytes)
    hasher.update(act_bytes)

    return hasher.hexdigest()


def compare_metrics(
    baseline: Dict[str, Any],
    resumed: Dict[str, Any],
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """
    Compare metrics between baseline and resumed runs.

    Returns comparison report with match/mismatch counts.
    """
    report = {
        'epsilon_matches': 0,
        'epsilon_mismatches': 0,
        'reward_matches': 0,
        'reward_mismatches': 0,
        'action_matches': 0,
        'action_mismatches': 0,
        'checksum_match': False,
        'max_epsilon_diff': 0.0,
        'max_reward_diff': 0.0,
        'total_steps': len(baseline['epsilons'])
    }

    # Compare epsilons
    for i, (eps1, eps2) in enumerate(zip(baseline['epsilons'], resumed['epsilons'])):
        diff = abs(eps1 - eps2)
        report['max_epsilon_diff'] = max(report['max_epsilon_diff'], diff)

        if diff < tolerance:
            report['epsilon_matches'] += 1
        else:
            report['epsilon_mismatches'] += 1

    # Compare rewards
    for i, (rew1, rew2) in enumerate(zip(baseline['rewards'], resumed['rewards'])):
        diff = abs(rew1 - rew2)
        report['max_reward_diff'] = max(report['max_reward_diff'], diff)

        if diff < tolerance:
            report['reward_matches'] += 1
        else:
            report['reward_mismatches'] += 1

    # Compare actions
    for i, (act1, act2) in enumerate(zip(baseline['actions'], resumed['actions'])):
        if act1 == act2:
            report['action_matches'] += 1
        else:
            report['action_mismatches'] += 1

    # Compare checksums
    report['checksum_match'] = baseline['checksum'] == resumed['checksum']
    report['baseline_checksum'] = baseline['checksum']
    report['resumed_checksum'] = resumed['checksum']

    return report


def print_comparison_report(report: Dict[str, Any]):
    """Print detailed comparison report."""
    print("\n" + "="*80)
    print("SAVE/RESUME DETERMINISM COMPARISON REPORT")
    print("="*80)

    total = report['total_steps']

    print(f"\nTotal Steps: {total}")
    print(f"\nChecksum Match: {'DONE PASS' if report['checksum_match'] else 'TODO FAIL'}")
    print(f"  Baseline:  {report['baseline_checksum'][:16]}...")
    print(f"  Resumed:   {report['resumed_checksum'][:16]}...")

    print(f"\nEpsilon Values:")
    print(f"  Matches:    {report['epsilon_matches']:5d} / {total} ({report['epsilon_matches']/total*100:.1f}%)")
    print(f"  Mismatches: {report['epsilon_mismatches']:5d} / {total} ({report['epsilon_mismatches']/total*100:.1f}%)")
    print(f"  Max Diff:   {report['max_epsilon_diff']:.2e}")

    print(f"\nRewards:")
    print(f"  Matches:    {report['reward_matches']:5d} / {total} ({report['reward_matches']/total*100:.1f}%)")
    print(f"  Mismatches: {report['reward_mismatches']:5d} / {total} ({report['reward_mismatches']/total*100:.1f}%)")
    print(f"  Max Diff:   {report['max_reward_diff']:.2e}")

    print(f"\nActions:")
    print(f"  Matches:    {report['action_matches']:5d} / {total} ({report['action_matches']/total*100:.1f}%)")
    print(f"  Mismatches: {report['action_mismatches']:5d} / {total} ({report['action_mismatches']/total*100:.1f}%)")

    print("\n" + "="*80)

    # Overall result
    all_match = (
        report['checksum_match'] and
        report['epsilon_mismatches'] == 0 and
        report['reward_mismatches'] == 0 and
        report['action_mismatches'] == 0
    )

    if all_match:
        print("RESULT: DONE PERFECT DETERMINISM - All metrics match exactly")
    else:
        print("RESULT: ⚠ PARTIAL DETERMINISM - Some mismatches detected")

    print("="*80 + "\n")


class TestSaveResumeDeterminism:
    """Test save/resume determinism with smoke test."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_smoke_test_10k_steps(self, temp_dir):
        """
        Smoke test: Run 5k steps, save checkpoint, resume, and verify determinism.

        This tests:
        1. Initial run produces consistent results
        2. Checkpoint save/load works
        3. Resumed run matches initial run exactly
        4. Epsilon, rewards, and actions are identical
        """
        print("\n" + "="*80)
        print("STARTING SAVE/RESUME DETERMINISM SMOKE TEST")
        print("="*80)

        # Configure determinism
        configure_determinism(enabled=True, strict=False)

        # Test parameters
        seed = 42
        num_actions = 6
        obs_shape = (4, 84, 84)
        checkpoint_step = 2500  # Save checkpoint halfway
        total_steps = 5000  # Total steps to test
        device = 'cpu'

        print(f"\nConfiguration:")
        print(f"  Seed: {seed}")
        print(f"  Total Steps: {total_steps}")
        print(f"  Checkpoint Step: {checkpoint_step}")
        print(f"  Device: {device}")

        # =====================================================================
        # Phase 1: Initial run to checkpoint
        # =====================================================================
        print(f"\n{'='*80}")
        print("PHASE 1: Initial run to checkpoint")
        print(f"{'='*80}")

        # Set seed
        set_seed(seed, deterministic=True)

        # Create environment
        env1 = SimpleMockEnv(obs_shape=obs_shape, num_actions=num_actions, seed=seed)

        # Create models
        online_model1 = DQN(num_actions=num_actions).to(device)
        target_model1 = DQN(num_actions=num_actions).to(device)
        target_model1.load_state_dict(online_model1.state_dict())

        # Create optimizer
        optimizer1 = configure_optimizer(online_model1, optimizer_type='rmsprop')

        # Create replay buffer
        replay_buffer1 = ReplayBuffer(capacity=10000, obs_shape=obs_shape)

        # Create epsilon scheduler
        epsilon_scheduler1 = EpsilonScheduler(
            epsilon_start=1.0,
            epsilon_end=0.1,
            decay_frames=1_000_000
        )

        # Run to checkpoint
        print(f"\nRunning {checkpoint_step} steps...")
        phase1_metrics = run_training_steps(
            env1, online_model1, target_model1, optimizer1,
            replay_buffer1, epsilon_scheduler1,
            num_steps=checkpoint_step,
            start_step=0,
            device=device
        )

        print(f"Completed. Epsilon: {phase1_metrics['epsilons'][-1]:.4f}")

        # Save checkpoint
        checkpoint_dir = Path(temp_dir) / 'checkpoints'
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

        checkpoint_path = manager.save_checkpoint(
            step=checkpoint_step,
            episode=100,  # Dummy episode count
            epsilon=epsilon_scheduler1.get_epsilon(checkpoint_step),
            online_model=online_model1,
            target_model=target_model1,
            optimizer=optimizer1,
            replay_buffer=replay_buffer1,
            rng_states=get_rng_states(env1),
            extra_metadata={'seed': seed}
        )

        print(f"\nCheckpoint saved: {checkpoint_path}")

        # =====================================================================
        # Phase 2: Continue from checkpoint (baseline)
        # =====================================================================
        print(f"\n{'='*80}")
        print("PHASE 2: Continue from checkpoint (baseline)")
        print(f"{'='*80}")

        remaining_steps = total_steps - checkpoint_step
        print(f"\nRunning {remaining_steps} more steps...")

        baseline_metrics = run_training_steps(
            env1, online_model1, target_model1, optimizer1,
            replay_buffer1, epsilon_scheduler1,
            num_steps=remaining_steps,
            start_step=checkpoint_step,
            device=device
        )

        print(f"Completed. Final epsilon: {baseline_metrics['epsilons'][-1]:.4f}")
        print(f"Baseline checksum: {baseline_metrics['checksum'][:16]}...")

        # =====================================================================
        # Phase 3: Resume from checkpoint
        # =====================================================================
        print(f"\n{'='*80}")
        print("PHASE 3: Resume from checkpoint and verify")
        print(f"{'='*80}")

        # Create fresh models for resume
        online_model2 = DQN(num_actions=num_actions).to(device)
        target_model2 = DQN(num_actions=num_actions).to(device)
        optimizer2 = configure_optimizer(online_model2, optimizer_type='rmsprop')
        epsilon_scheduler2 = EpsilonScheduler(
            epsilon_start=1.0,
            epsilon_end=0.1,
            decay_frames=1_000_000
        )
        replay_buffer2 = ReplayBuffer(capacity=10000, obs_shape=obs_shape)
        env2 = SimpleMockEnv(obs_shape=obs_shape, num_actions=num_actions, seed=seed)

        # Resume from checkpoint
        print(f"\nResuming from checkpoint...")
        resumed_state = resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=online_model2,
            target_model=target_model2,
            optimizer=optimizer2,
            epsilon_scheduler=epsilon_scheduler2,
            replay_buffer=replay_buffer2,
            env=env2,
            device=device
        )

        print(f"Resumed from step {resumed_state['step']}")

        # Run same number of steps
        print(f"\nRunning {remaining_steps} steps after resume...")
        resumed_metrics = run_training_steps(
            env2, online_model2, target_model2, optimizer2,
            replay_buffer2, epsilon_scheduler2,
            num_steps=remaining_steps,
            start_step=checkpoint_step,
            device=device
        )

        print(f"Completed. Final epsilon: {resumed_metrics['epsilons'][-1]:.4f}")
        print(f"Resumed checksum: {resumed_metrics['checksum'][:16]}...")

        # =====================================================================
        # Phase 4: Compare and report
        # =====================================================================
        print(f"\n{'='*80}")
        print("PHASE 4: Compare baseline vs resumed")
        print(f"{'='*80}")

        report = compare_metrics(baseline_metrics, resumed_metrics, tolerance=1e-5)
        print_comparison_report(report)

        # =====================================================================
        # Assertions
        # =====================================================================

        # Checksums should match
        assert report['checksum_match'], \
            f"Checksum mismatch: {baseline_metrics['checksum']} != {resumed_metrics['checksum']}"

        # All epsilons should match (within tolerance)
        assert report['epsilon_mismatches'] == 0, \
            f"Epsilon mismatches detected: {report['epsilon_mismatches']} / {report['total_steps']}"

        # All rewards should match (within tolerance)
        assert report['reward_mismatches'] == 0, \
            f"Reward mismatches detected: {report['reward_mismatches']} / {report['total_steps']}"

        # All actions should match exactly
        assert report['action_mismatches'] == 0, \
            f"Action mismatches detected: {report['action_mismatches']} / {report['total_steps']}"

        print("\nDONE All assertions passed - Determinism verified!\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
