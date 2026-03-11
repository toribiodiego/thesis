"""
Tests for the distributional RL loss (C51 projection and cross-entropy).

Verifies:
- Projection places probability on correct atoms (hand-computed)
- Boundary clipping to [v_min, v_max]
- Terminal states collapse distribution to reward atom
- Projected distribution sums to 1
- Cross-entropy values match manual computation
- Per-sample loss shape correct for priority updates
- Multi-step targets use gamma^n
- Gradients flow through online log-probs only
"""

import numpy as np
import pytest
import torch

from src.training.distributional import (
    compute_distributional_loss,
    project_distribution,
)


# 5-atom support [-2, -1, 0, 1, 2] with delta_z = 1
SUPPORT = torch.linspace(-2.0, 2.0, 5)


# ---------------------------------------------------------------------------
# Projection: known-answer tests
# ---------------------------------------------------------------------------


class TestProjectDistribution:
    """C51 Algorithm 1 projection with hand-computed expected values."""

    def test_identity_projection(self):
        """r=0, gamma=1, done=False -> Tz = z (no shift).

        Each atom maps to itself, so projected == next_probs.
        """
        next_probs = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])
        rewards = torch.tensor([0.0])
        dones = torch.tensor([False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=1.0)

        torch.testing.assert_close(projected, next_probs, atol=1e-6, rtol=0)

    def test_shift_by_half_atom(self):
        """r=0.5, gamma=1, done=False with point mass on z=0.

        Tz for atom 2 (z=0): 0.5 + 0 = 0.5, between atoms 2 and 3.
        b = (0.5+2)/1 = 2.5 -> l=2, u=3, weights 0.5/0.5.
        Expected: [0, 0, 0.5, 0.5, 0].
        """
        next_probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        rewards = torch.tensor([0.5])
        dones = torch.tensor([False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=1.0)

        expected = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]])
        torch.testing.assert_close(projected, expected, atol=1e-6, rtol=0)

    def test_contraction_with_gamma(self):
        """r=0, gamma=0.5, done=False contracts distribution toward 0.

        next_probs = [0, 0.5, 0, 0.5, 0] (atoms at z=-1 and z=1).
        Tz(-1) = 0.5*(-1) = -0.5, b=1.5: l=1,u=2, wt 0.5/0.5
        Tz(1)  = 0.5*(1)  =  0.5, b=2.5: l=2,u=3, wt 0.5/0.5
        Expected: [0, 0.25, 0.5, 0.25, 0].
        """
        next_probs = torch.tensor([[0.0, 0.5, 0.0, 0.5, 0.0]])
        rewards = torch.tensor([0.0])
        dones = torch.tensor([False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=0.5)

        expected = torch.tensor([[0.0, 0.25, 0.5, 0.25, 0.0]])
        torch.testing.assert_close(projected, expected, atol=1e-6, rtol=0)

    def test_terminal_state_exact_atom(self):
        """done=True, r=1.0 -> Tz=1.0 for all atoms -> atom 3.

        All probability collapses to the atom at z=1.
        """
        next_probs = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])
        rewards = torch.tensor([1.0])
        dones = torch.tensor([True])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=1.0)

        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
        torch.testing.assert_close(projected, expected, atol=1e-6, rtol=0)

    def test_terminal_state_between_atoms(self):
        """done=True, r=0.3 -> Tz=0.3 for all atoms.

        b = (0.3+2)/1 = 2.3, l=2, u=3, wt_l=0.7, wt_u=0.3.
        All probability splits between atoms 2 and 3.
        """
        next_probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        rewards = torch.tensor([0.3])
        dones = torch.tensor([True])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=1.0)

        expected = torch.tensor([[0.0, 0.0, 0.7, 0.3, 0.0]])
        torch.testing.assert_close(projected, expected, atol=1e-6, rtol=0)

    def test_projected_sums_to_one(self):
        """Projected distribution always sums to 1."""
        next_probs = torch.tensor([[0.15, 0.25, 0.3, 0.2, 0.1]])
        rewards = torch.tensor([0.7])
        dones = torch.tensor([False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=0.8)

        assert abs(projected.sum().item() - 1.0) < 1e-6

    def test_batch_projected_sums_to_one(self):
        """Each sample in a batch sums to 1."""
        next_probs = torch.rand(8, 5)
        next_probs = next_probs / next_probs.sum(dim=1, keepdim=True)
        rewards = torch.randn(8)
        dones = torch.tensor([False, True, False, True, False, False, True, False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=0.99)

        sums = projected.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Boundary clipping
# ---------------------------------------------------------------------------


class TestBoundaryClipping:
    """Projection clips shifted atoms to [v_min, v_max]."""

    def test_clip_above_v_max(self):
        """r=5, gamma=1 -> all Tz values exceed v_max=2.

        Tz = 5 + z = [3, 4, 5, 6, 7], all clipped to 2.
        All probability goes to the last atom.
        """
        next_probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        rewards = torch.tensor([5.0])
        dones = torch.tensor([False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=1.0)

        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        torch.testing.assert_close(projected, expected, atol=1e-6, rtol=0)

    def test_clip_below_v_min(self):
        """r=-5, gamma=1 -> all Tz values below v_min=-2.

        Tz = -5 + z = [-7, -6, -5, -4, -3], all clipped to -2.
        All probability goes to the first atom.
        """
        next_probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        rewards = torch.tensor([-5.0])
        dones = torch.tensor([False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=1.0)

        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(projected, expected, atol=1e-6, rtol=0)

    def test_partial_clipping(self):
        """r=1.5, gamma=1 -> some atoms clipped, some not.

        Tz = 1.5 + z = [-0.5, 0.5, 1.5, 2.5, 3.5]
        Clipped: [-0.5, 0.5, 1.5, 2.0, 2.0]
        Point mass on atom 0 (z=-2):
        Tz = -0.5, b = (-0.5+2)/1 = 1.5
        l=1, u=2, wt_l=0.5, wt_u=0.5
        Expected: [0, 0.5, 0.5, 0, 0]
        """
        next_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        rewards = torch.tensor([1.5])
        dones = torch.tensor([False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=1.0)

        expected = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]])
        torch.testing.assert_close(projected, expected, atol=1e-6, rtol=0)

    def test_all_mass_clipped_to_same_atom(self):
        """When all Tz values clip to the same atom, projection is valid."""
        next_probs = torch.tensor([[0.3, 0.3, 0.4, 0.0, 0.0]])
        rewards = torch.tensor([10.0])
        dones = torch.tensor([False])

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, gamma=1.0)

        # All clipped to v_max, all mass on last atom
        assert abs(projected[0, -1].item() - 1.0) < 1e-6
        assert abs(projected.sum().item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Multi-step
# ---------------------------------------------------------------------------


class TestMultiStep:
    """Multi-step uses gamma^n, producing different projections."""

    def test_gamma_n_vs_gamma(self):
        """gamma^3 != gamma, so projections differ.

        Point mass on z=2, r=0, done=False.
        gamma=0.99: Tz = 0.99*2 = 1.98
        gamma^3=0.970299: Tz = 0.970299*2 = 1.940598
        """
        next_probs = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        rewards = torch.tensor([0.0])
        dones = torch.tensor([False])

        proj_1step = project_distribution(
            next_probs, rewards, dones, SUPPORT, gamma=0.99,
        )
        proj_3step = project_distribution(
            next_probs, rewards, dones, SUPPORT, gamma=0.99 ** 3,
        )

        assert not torch.allclose(proj_1step, proj_3step, atol=1e-4)

    def test_gamma_n_correct_values(self):
        """Hand-computed 3-step projection.

        gamma^3 = 0.970299, r=0, point mass on z=2.
        Tz = 0.970299*2 = 1.940598
        b = (1.940598+2)/1 = 3.940598
        l=3, u=4, wt_l = 4-3.940598 = 0.059402, wt_u = 0.940598
        """
        next_probs = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        rewards = torch.tensor([0.0])
        dones = torch.tensor([False])
        gamma_n = 0.99 ** 3  # 0.970299

        projected = project_distribution(
            next_probs, rewards, dones, SUPPORT, gamma=gamma_n,
        )

        expected = torch.tensor([[0.0, 0.0, 0.0, 0.059402, 0.940598]])
        torch.testing.assert_close(projected, expected, atol=1e-4, rtol=0)

    def test_gamma_n_with_reward(self):
        """Multi-step with nonzero n-step return.

        gamma^3 = 0.970299, R^(3)=2.5, point mass on z=0, done=False.
        Tz = 2.5 + 0.970299*0 = 2.5, clipped to 2.0.
        b = (2+2)/1 = 4. All mass on atom 4.
        """
        next_probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        rewards = torch.tensor([2.5])
        dones = torch.tensor([False])

        projected = project_distribution(
            next_probs, rewards, dones, SUPPORT, gamma=0.99 ** 3,
        )

        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        torch.testing.assert_close(projected, expected, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Cross-entropy loss
# ---------------------------------------------------------------------------


def _make_loss_inputs(
    online_probs, target_probs, actions, next_actions, rewards, dones,
    num_actions=2,
):
    """Build tensors for compute_distributional_loss from probability arrays.

    Constructs (B, A, atoms) log-prob tensors from per-sample probability
    vectors, filling unused actions with uniform distributions.
    """
    batch_size = len(rewards)
    num_atoms = SUPPORT.shape[0]

    # Build online log-probs (B, A, atoms)
    online_lp = torch.full(
        (batch_size, num_actions, num_atoms), 1.0 / num_atoms,
    ).log()
    for i in range(batch_size):
        a = actions[i]
        online_lp[i, a] = torch.tensor(online_probs[i]).log()
    online_lp = online_lp.clone().detach().requires_grad_(True)

    # Build target log-probs (B, A, atoms) -- detached
    target_lp = torch.full(
        (batch_size, num_actions, num_atoms), 1.0 / num_atoms,
    ).log()
    for i in range(batch_size):
        a = next_actions[i]
        target_lp[i, a] = torch.tensor(target_probs[i]).log()

    return {
        "online_log_probs": online_lp,
        "actions": torch.tensor(actions, dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "dones": torch.tensor(dones),
        "target_log_probs": target_lp,
        "next_actions": torch.tensor(next_actions, dtype=torch.long),
        "support": SUPPORT,
        "gamma": 1.0,
    }


class TestDistributionalLoss:
    """Cross-entropy loss computation with known answers."""

    def test_output_keys(self):
        inputs = _make_loss_inputs(
            online_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]],
            target_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]],
            actions=[0], next_actions=[0],
            rewards=[0.0], dones=[False],
        )
        result = compute_distributional_loss(**inputs)
        assert set(result.keys()) == {"loss", "per_sample_loss"}

    def test_loss_is_scalar(self):
        inputs = _make_loss_inputs(
            online_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]],
            target_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]],
            actions=[0], next_actions=[0],
            rewards=[0.0], dones=[False],
        )
        result = compute_distributional_loss(**inputs)
        assert result["loss"].dim() == 0

    def test_per_sample_loss_shape(self):
        """per_sample_loss has shape (B,) for priority updates."""
        inputs = _make_loss_inputs(
            online_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]] * 4,
            target_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]] * 4,
            actions=[0, 1, 0, 1], next_actions=[0, 0, 1, 1],
            rewards=[0.0, 0.5, -0.5, 1.0], dones=[False, False, False, True],
        )
        result = compute_distributional_loss(**inputs)
        assert result["per_sample_loss"].shape == (4,)

    def test_cross_entropy_identity_projection(self):
        """r=0, gamma=1, done=False -> identity projection.

        target = [0.1, 0.2, 0.4, 0.2, 0.1] (no shift)
        online = [0.05, 0.1, 0.5, 0.25, 0.1]
        CE = -(0.1*log(0.05) + 0.2*log(0.1) + 0.4*log(0.5)
               + 0.2*log(0.25) + 0.1*log(0.1))
           = 1.54487
        """
        inputs = _make_loss_inputs(
            online_probs=[[0.05, 0.1, 0.5, 0.25, 0.1]],
            target_probs=[[0.1, 0.2, 0.4, 0.2, 0.1]],
            actions=[0], next_actions=[1],
            rewards=[0.0], dones=[False],
        )
        result = compute_distributional_loss(**inputs)

        expected_ce = -(
            0.1 * np.log(0.05) + 0.2 * np.log(0.1) + 0.4 * np.log(0.5)
            + 0.2 * np.log(0.25) + 0.1 * np.log(0.1)
        )
        assert abs(result["loss"].item() - expected_ce) < 1e-4

    def test_cross_entropy_terminal(self):
        """done=True, r=1.0 -> all mass on atom 3 (z=1).

        projected = [0, 0, 0, 1, 0]
        online = [0.05, 0.1, 0.5, 0.25, 0.1]
        CE = -(1 * log(0.25)) = -log(0.25) = log(4) = 1.38629
        """
        inputs = _make_loss_inputs(
            online_probs=[[0.05, 0.1, 0.5, 0.25, 0.1]],
            target_probs=[[0.1, 0.2, 0.4, 0.2, 0.1]],
            actions=[0], next_actions=[1],
            rewards=[1.0], dones=[True],
        )
        result = compute_distributional_loss(**inputs)

        expected_ce = -np.log(0.25)  # 1.38629
        assert abs(result["loss"].item() - expected_ce) < 1e-4

    def test_batch_mean_loss(self):
        """Mean loss across batch matches average of per-sample values.

        Sample 0: identity projection, CE = 1.54487
        Sample 1: terminal r=1, CE = 1.38629
        Mean = (1.54487 + 1.38629) / 2 = 1.46558
        """
        inputs = _make_loss_inputs(
            online_probs=[
                [0.05, 0.1, 0.5, 0.25, 0.1],
                [0.05, 0.1, 0.5, 0.25, 0.1],
            ],
            target_probs=[
                [0.1, 0.2, 0.4, 0.2, 0.1],
                [0.1, 0.2, 0.4, 0.2, 0.1],
            ],
            actions=[0, 0], next_actions=[1, 1],
            rewards=[0.0, 1.0], dones=[False, True],
        )
        result = compute_distributional_loss(**inputs)

        # Verify mean = average of per-sample
        torch.testing.assert_close(
            result["loss"],
            result["per_sample_loss"].mean(),
            atol=1e-6, rtol=0,
        )

        # Verify against hand-computed values
        ce_0 = -(
            0.1 * np.log(0.05) + 0.2 * np.log(0.1) + 0.4 * np.log(0.5)
            + 0.2 * np.log(0.25) + 0.1 * np.log(0.1)
        )
        ce_1 = -np.log(0.25)
        expected_mean = (ce_0 + ce_1) / 2

        assert abs(result["loss"].item() - expected_mean) < 1e-4

    def test_uniform_online_gives_log_n_atoms(self):
        """Uniform online + any valid projected -> CE = -sum(m_i * log(1/N)).

        For N=5: CE = -log(1/5) = log(5) = 1.6094.
        """
        inputs = _make_loss_inputs(
            online_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]],
            target_probs=[[0.3, 0.3, 0.2, 0.1, 0.1]],
            actions=[0], next_actions=[0],
            rewards=[0.0], dones=[False],
        )
        result = compute_distributional_loss(**inputs)

        expected_ce = np.log(5)  # 1.6094
        assert abs(result["loss"].item() - expected_ce) < 1e-4

    def test_perfect_match_gives_entropy(self):
        """When online == projected, CE = entropy of the distribution.

        target = online = [0.1, 0.2, 0.4, 0.2, 0.1], r=0, gamma=1
        CE = entropy = -sum(p * log(p))
        """
        probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        inputs = _make_loss_inputs(
            online_probs=[probs],
            target_probs=[probs],
            actions=[0], next_actions=[1],
            rewards=[0.0], dones=[False],
        )
        result = compute_distributional_loss(**inputs)

        entropy = -sum(p * np.log(p) for p in probs)  # 1.4708
        assert abs(result["loss"].item() - entropy) < 1e-4


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Gradients flow through online log-probs only."""

    def test_loss_backward(self):
        """loss.backward() computes gradients for online_log_probs."""
        inputs = _make_loss_inputs(
            online_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]],
            target_probs=[[0.1, 0.3, 0.3, 0.2, 0.1]],
            actions=[0], next_actions=[0],
            rewards=[0.5], dones=[False],
        )
        result = compute_distributional_loss(**inputs)
        result["loss"].backward()

        assert inputs["online_log_probs"].grad is not None
        assert inputs["online_log_probs"].grad.abs().sum() > 0

    def test_per_sample_loss_has_gradients(self):
        """per_sample_loss retains gradient graph for IS weighting."""
        inputs = _make_loss_inputs(
            online_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]] * 3,
            target_probs=[[0.1, 0.3, 0.3, 0.2, 0.1]] * 3,
            actions=[0, 0, 0], next_actions=[0, 0, 0],
            rewards=[0.0, 0.5, 1.0], dones=[False, False, True],
        )
        result = compute_distributional_loss(**inputs)

        # Simulate IS weighting: (weights * per_sample_loss).mean()
        is_weights = torch.tensor([1.0, 0.5, 0.8])
        weighted_loss = (is_weights * result["per_sample_loss"]).mean()
        weighted_loss.backward()

        assert inputs["online_log_probs"].grad is not None
        assert inputs["online_log_probs"].grad.abs().sum() > 0

    def test_no_gradient_through_target(self):
        """Target log-probs should not receive gradients."""
        inputs = _make_loss_inputs(
            online_probs=[[0.2, 0.2, 0.2, 0.2, 0.2]],
            target_probs=[[0.3, 0.3, 0.2, 0.1, 0.1]],
            actions=[0], next_actions=[0],
            rewards=[0.0], dones=[False],
        )
        # Target log-probs are not leaves with requires_grad, so
        # backward should not fail and should not affect target
        result = compute_distributional_loss(**inputs)
        result["loss"].backward()

        assert not inputs["target_log_probs"].requires_grad

    def test_loss_is_nonnegative(self):
        """Cross-entropy between valid distributions is non-negative."""
        inputs = _make_loss_inputs(
            online_probs=[[0.05, 0.1, 0.5, 0.25, 0.1]],
            target_probs=[[0.3, 0.3, 0.2, 0.1, 0.1]],
            actions=[0], next_actions=[0],
            rewards=[0.3], dones=[False],
        )
        result = compute_distributional_loss(**inputs)
        assert result["loss"].item() >= 0

    def test_no_nans(self):
        """Output should not contain NaN values."""
        inputs = _make_loss_inputs(
            online_probs=[[0.05, 0.1, 0.5, 0.25, 0.1]] * 4,
            target_probs=[[0.3, 0.1, 0.1, 0.3, 0.2]] * 4,
            actions=[0, 1, 0, 1], next_actions=[1, 0, 1, 0],
            rewards=[0.0, 1.0, -1.0, 0.5],
            dones=[False, True, False, False],
        )
        result = compute_distributional_loss(**inputs)

        assert not torch.isnan(result["loss"])
        assert not torch.isnan(result["per_sample_loss"]).any()


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


class TestActionSelection:
    """Loss selects correct action distributions from (B, A, atoms)."""

    def test_different_actions_give_different_losses(self):
        """Selecting different online actions produces different losses."""
        # Two actions with different online distributions
        online_lp = torch.log(torch.tensor([
            [[0.05, 0.1, 0.5, 0.25, 0.1],
             [0.3, 0.3, 0.2, 0.1, 0.1]],
        ])).requires_grad_(True)

        target_lp = torch.log(torch.tensor([
            [[0.2, 0.2, 0.2, 0.2, 0.2],
             [0.1, 0.2, 0.4, 0.2, 0.1]],
        ]))

        result_a0 = compute_distributional_loss(
            online_lp, torch.tensor([0]), torch.tensor([0.0]),
            torch.tensor([False]), target_lp, torch.tensor([1]),
            SUPPORT, gamma=1.0,
        )
        result_a1 = compute_distributional_loss(
            online_lp.detach().requires_grad_(True),
            torch.tensor([1]), torch.tensor([0.0]),
            torch.tensor([False]), target_lp, torch.tensor([1]),
            SUPPORT, gamma=1.0,
        )

        assert result_a0["loss"].item() != result_a1["loss"].item()

    def test_different_next_actions_give_different_targets(self):
        """Selecting different target actions produces different losses."""
        online_lp = torch.log(torch.tensor([
            [[0.2, 0.2, 0.2, 0.2, 0.2],
             [0.2, 0.2, 0.2, 0.2, 0.2]],
        ])).requires_grad_(True)

        # Two different target distributions for the two actions
        target_lp = torch.log(torch.tensor([
            [[0.1, 0.2, 0.4, 0.2, 0.1],
             [0.4, 0.1, 0.1, 0.1, 0.3]],
        ]))

        result_na0 = compute_distributional_loss(
            online_lp, torch.tensor([0]), torch.tensor([0.5]),
            torch.tensor([False]), target_lp, torch.tensor([0]),
            SUPPORT, gamma=1.0,
        )
        result_na1 = compute_distributional_loss(
            online_lp.detach().requires_grad_(True),
            torch.tensor([0]), torch.tensor([0.5]),
            torch.tensor([False]), target_lp, torch.tensor([1]),
            SUPPORT, gamma=1.0,
        )

        assert result_na0["loss"].item() != result_na1["loss"].item()
