# Episode Handling in DQN Training

Comprehensive guide to episode management, termination policies, life-loss handling, no-op starts, and differences between training and evaluation modes.

## Table of Contents

1. [Overview](#overview)
2. [Episode Termination Policies](#episode-termination-policies)
3. [Life-Loss as Terminal (Training Mode)](#life-loss-as-terminal-training-mode)
4. [Full Episodes (Evaluation Mode)](#full-episodes-evaluation-mode)
5. [No-Op Starts](#no-op-starts)
6. [Episode Tracking and Metrics](#episode-tracking-and-metrics)
7. [Implementation Guide](#implementation-guide)
8. [Configuration](#configuration)
9. [Common Pitfalls](#common-pitfalls)

---

## Overview

Episode handling is a critical aspect of DQN training that affects both learning efficiency and evaluation accuracy. The key challenge is balancing two competing objectives:

1. **Training Efficiency**: Want agent to learn quickly from meaningful experiences
2. **Evaluation Accuracy**: Need to measure true performance without training biases

The 2015 Nature DQN paper introduced several episode handling techniques that differ between training and evaluation:

- **Training**: Treat life-loss as terminal to enable faster learning
- **Evaluation**: Run full episodes (all lives) to measure true performance
- **Both**: Use no-op starts for stochasticity and episode reset handling

---

## Episode Termination Policies

### Standard Gymnasium Termination

Gymnasium environments return two termination flags:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

- **`terminated`**: Episode ended naturally (game over, goal reached, etc.)
- **`truncated`**: Episode ended artificially (time limit, wrapper constraint, etc.)

**Combined done flag:**
```python
done = terminated or truncated
```

### DQN Termination Behavior

For DQN training, we treat both termination types the same:

```python
# Store in replay buffer
replay_buffer.append(state, action, reward, next_state, done=terminated or truncated)

# TD target computation
td_target = reward + gamma * (1 - done) * max_Q_target(next_state)
```

When `done=True`:
- TD target becomes just `reward` (no bootstrap from next state)
- Episode must be reset before next step
- Episode metrics (return, length) are recorded

---

## Life-Loss as Terminal (Training Mode)

### Motivation

Atari games with lives (e.g., Breakout, Space Invaders) naturally span multiple sub-episodes within a single game. Treating life-loss as terminal during training provides:

1. **Faster Learning**: Agent learns consequences of death immediately
2. **Better Credit Assignment**: Negative feedback is more immediate
3. **Improved Sample Efficiency**: More terminal states → more bootstrap cutoffs

### Implementation with EpisodicLifeEnv Wrapper

The `EpisodicLifeEnv` wrapper (from `stable-baselines3` or custom) modifies the environment:

```python
from gymnasium.wrappers import EpisodicLifeEnv

# Wrap environment for training
train_env = make_atari_env(game_id='BreakoutNoFrameskip-v4')
train_env = EpisodicLifeEnv(train_env)  # Life-loss becomes terminal
```

**Behavior:**
- When agent loses a life, `terminated=True` is returned
- Environment internally tracks real game over vs. life-loss
- On life-loss: state continues from current game (no full reset)
- On real game over: full reset occurs

**Example:**
```python
# Agent has 5 lives in Breakout

# Life 1: Play until first death
obs, reward, terminated, truncated, info = env.step(action)
# terminated=True (life lost), lives_remaining=4

# Environment auto-continues (no manual reset needed for life-loss)
obs, reward, terminated, truncated, info = env.step(next_action)
# Agent continues with 4 lives

# ... repeat until all lives lost ...

# Final life lost: real game over
obs, reward, terminated, truncated, info = env.step(action)
# terminated=True (game over), info['episode'] contains episode stats

# Now must reset manually
obs, info = env.reset()
```

**TD Target Impact:**
```python
# When life is lost (terminated=True):
td_target = reward + 0.99 * (1 - 1.0) * max_Q = reward
# Agent learns: "this state leads to immediate end (negative reward)"

# Regular step (terminated=False):
td_target = reward + 0.99 * (1 - 0.0) * max_Q = reward + 0.99 * max_Q
# Agent learns: "this state has future value"
```

### When to Use

**Use EpisodicLifeEnv during training:**
- Atari games with lives (Breakout, Space Invaders, Seaquest, etc.)
- Want faster convergence with better credit assignment
- Following 2015 Nature DQN paper methodology

**Do NOT use:**
- Games without lives (Pong has "lives" but they're just score tracking)
- Evaluation mode (want true episode returns)
- Single-life games

---

## Full Episodes (Evaluation Mode)

### Motivation

Evaluation should measure the agent's true performance across complete episodes without training artifacts. This means:

1. **No Life-Loss Terminal**: Run all lives in a single episode
2. **No Learning**: Disable gradient computation and optimizer updates
3. **Low/Fixed Epsilon**: Use eval_epsilon (e.g., 0.05) or greedy (ε=0)
4. **Consistent Seeds**: Use deterministic environment for reproducibility

### Implementation

```python
def evaluate(agent, env_id, num_episodes=10, eval_epsilon=0.05, seed=None):
    """
    Evaluate agent over multiple full episodes.

    Args:
        agent: DQN agent with policy network
        env_id: Environment identifier (e.g., 'BreakoutNoFrameskip-v4')
        num_episodes: Number of episodes to run
        eval_epsilon: Exploration rate (0.05 = 5% random, 0.0 = greedy)
        seed: Random seed for reproducibility

    Returns:
        dict: Episode statistics (mean/median/std returns, lengths)
    """
    # Create evaluation environment (NO EpisodicLifeEnv wrapper)
    eval_env = make_atari_env(env_id)
    if seed is not None:
        eval_env.reset(seed=seed)

    # Set agent to eval mode
    agent.eval()

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Select action with eval epsilon
            with torch.no_grad():
                if random.random() < eval_epsilon:
                    action = eval_env.action_space.sample()
                else:
                    # Greedy action
                    q_values = agent(obs)
                    action = q_values.argmax().item()

            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    return {
        'mean_return': np.mean(episode_returns),
        'median_return': np.median(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'episodes': num_episodes
    }
```

**Key Differences from Training:**

| Aspect | Training Mode | Evaluation Mode |
|--------|---------------|-----------------|
| Life-loss | Terminal (EpisodicLifeEnv) | Not terminal (full episodes) |
| Epsilon | Decaying (1.0→0.1) | Fixed low (0.05) or greedy (0.0) |
| Learning | Enabled (gradient updates) | Disabled (`eval()`, `no_grad()`) |
| Replay buffer | Active (append transitions) | Not used |
| Stochasticity | High (exploration) | Low (mostly greedy) |
| Episode length | Variable (life-based) | Full (all lives) |

---

## No-Op Starts

### Motivation

Atari games are deterministic given the same input sequence. To prevent overfitting to specific starting states, the DQN paper uses **no-op starts**:

1. At episode reset, perform a random number of "do nothing" (no-op) actions
2. Range: 1 to 30 no-op actions (uniformly sampled)
3. Provides stochastic starting states
4. Forces agent to generalize across different initial conditions

### Implementation with NoopResetEnv Wrapper

```python
from stable_baselines3.common.atari_wrappers import NoopResetEnv

# Wrap environment with no-op starts
env = make_atari_env(game_id)
env = NoopResetEnv(env, noop_max=30)  # Up to 30 no-ops on reset
```

**Behavior:**
```python
obs, info = env.reset()
# Internally:
# 1. Call base_env.reset()
# 2. Sample n ~ Uniform(1, 30)
# 3. Execute no-op action n times
# 4. Return final observation
```

**Benefits:**
- Breaks determinism (same policy won't see same states every episode)
- Improves generalization (agent sees diverse starting positions)
- Matches DQN paper methodology

**Caveats:**
- Adds 1-30 frames to episode start (affects frame counting)
- Can cause early termination if game has time limits
- Evaluation should use same no-op policy for consistency

### Configuration

```yaml
# Training config
env:
  noop_max: 30  # Standard DQN value

# Evaluation config (optional: disable for reproducibility)
eval:
  noop_max: 0   # Deterministic starts
  # OR
  noop_max: 30  # Match training (better reflects real performance)
```

**Recommendation:** Use `noop_max=30` for both training and evaluation to match the DQN paper.

---

## Episode Tracking and Metrics

### Per-Episode Statistics

Track the following metrics for each completed episode:

```python
class EpisodeTracker:
    """Track episode statistics during training."""

    def __init__(self):
        self.current_return = 0.0
        self.current_length = 0
        self.episode_count = 0
        self.episode_returns = []
        self.episode_lengths = []

    def step(self, reward, done):
        """Update episode stats after each environment step."""
        self.current_return += reward
        self.current_length += 1

        if done:
            # Episode finished
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            self.episode_count += 1

            # Reset for next episode
            self.current_return = 0.0
            self.current_length = 0

            return {
                'episode_return': self.episode_returns[-1],
                'episode_length': self.episode_lengths[-1],
                'episode_count': self.episode_count
            }
        return None

    def get_recent_stats(self, n=100):
        """Get statistics over last n episodes."""
        recent_returns = self.episode_returns[-n:]
        recent_lengths = self.episode_lengths[-n:]

        return {
            'mean_return': np.mean(recent_returns),
            'std_return': np.std(recent_returns),
            'mean_length': np.mean(recent_lengths),
            'num_episodes': len(recent_returns)
        }
```

### Integration with Training Loop

```python
episode_tracker = EpisodeTracker()

for step in range(total_steps):
    # Training step
    result = training_step(...)

    # Track episode
    episode_stats = episode_tracker.step(result['reward'],
                                         result['terminated'] or result['truncated'])

    if episode_stats is not None:
        # Episode completed - log metrics
        logger.log_episode(
            step=step,
            return=episode_stats['episode_return'],
            length=episode_stats['episode_length'],
            epsilon=result['epsilon'],
            recent_mean=episode_tracker.get_recent_stats(100)['mean_return']
        )

        # Reset environment
        obs, info = env.reset()
```

### Important Episode Metrics

**Training Metrics:**
- `episode_return`: Total undiscounted reward (sum of all rewards in episode)
- `episode_length`: Number of steps in episode
- `rolling_mean_return`: Mean return over last N episodes (e.g., N=100)
- `episodes_per_frame`: Episode completion rate (higher = shorter episodes)

**Evaluation Metrics:**
- `mean_eval_return`: Average return over evaluation episodes
- `median_eval_return`: Median return (more robust to outliers)
- `std_eval_return`: Standard deviation (measures consistency)
- `min/max_eval_return`: Range of performance

---

## Implementation Guide

### Complete Training Setup with Episode Handling

```python
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv,
    FireResetEnv, ClipRewardEnv
)

def make_training_env(env_id, noop_max=30, frameskip=4):
    """
    Create training environment with all DQN wrappers.

    Includes:
    - No-op starts for stochasticity
    - Frame skipping for efficiency
    - Life-loss as terminal for faster learning
    - Fire action on reset (for games that require it)
    - Reward clipping to [-1, +1]
    """
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frameskip)
    env = EpisodicLifeEnv(env)  # Life-loss = terminal

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = ClipRewardEnv(env)  # Clip rewards to {-1, 0, +1}

    return env

def make_eval_env(env_id, noop_max=30, frameskip=4):
    """
    Create evaluation environment (NO life-loss wrapper).

    Differs from training:
    - NO EpisodicLifeEnv (full episodes)
    - Same other wrappers for consistency
    """
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frameskip)
    # NOTE: No EpisodicLifeEnv here!

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = ClipRewardEnv(env)

    return env
```

### Episode Reset Logic

```python
# Training loop with proper reset handling
obs, info = env.reset(seed=seed)
episode_tracker = EpisodeTracker()

for step in range(total_steps):
    # Select and execute action
    result = training_step(
        env=env,
        online_net=online_net,
        state=obs,
        # ... other params
    )

    # Update episode tracker
    episode_stats = episode_tracker.step(
        reward=result['reward'],
        done=result['terminated'] or result['truncated']
    )

    # Handle episode end
    if episode_stats is not None:
        # Log episode metrics
        print(f"Episode {episode_stats['episode_count']}: "
              f"Return={episode_stats['episode_return']:.1f}, "
              f"Length={episode_stats['episode_length']}")

        # Reset environment
        obs, info = env.reset()
    else:
        # Continue episode
        obs = result['next_state']
```

---

## Configuration

### YAML Configuration Example

```yaml
# experiments/dqn_atari/configs/base.yaml

env:
  # No-op starts (DQN paper default)
  noop_max: 30

  # Frame skip (4 for most games)
  frameskip: 4

  # Life-loss handling
  episodic_life: true  # Treat life-loss as terminal during training

training:
  # Episode tracking
  log_episode_stats: true
  rolling_window: 100  # Compute rolling mean over last 100 episodes

eval:
  # Evaluation settings
  num_episodes: 10
  eval_epsilon: 0.05  # Small exploration during eval
  episodic_life: false  # Full episodes (no life-loss terminal)
  noop_max: 30  # Match training or set to 0 for determinism
```

### Environment Creation from Config

```python
def make_env_from_config(config, mode='train'):
    """Create environment based on config and mode."""
    env_cfg = config['env']

    env = gym.make(env_cfg['env_id'])
    env = NoopResetEnv(env, noop_max=env_cfg['noop_max'])
    env = MaxAndSkipEnv(env, skip=env_cfg['frameskip'])

    # Apply life-loss wrapper based on mode
    if mode == 'train' and env_cfg.get('episodic_life', True):
        env = EpisodicLifeEnv(env)
    elif mode == 'eval' and config['eval'].get('episodic_life', False):
        env = EpisodicLifeEnv(env)  # Usually False for eval

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = ClipRewardEnv(env)

    return env
```

---

## Common Pitfalls

### 1. Forgetting to Reset After Episode End

**Problem:**
```python
# WRONG: No reset after done
while step < max_steps:
    result = training_step(env, ...)
    if result['terminated'] or result['truncated']:
        # Missing: obs, info = env.reset()
        pass
    obs = result['next_state']  # BUG: obs is invalid after episode end
```

**Solution:**
```python
# CORRECT: Reset on episode end
while step < max_steps:
    result = training_step(env, ...)
    if result['terminated'] or result['truncated']:
        obs, info = env.reset()  # Fresh start
    else:
        obs = result['next_state']
```

### 2. Using EpisodicLifeEnv During Evaluation

**Problem:** Training returns won't match evaluation returns because episodes have different lengths.

**Solution:** Create separate train/eval environments:
```python
train_env = make_training_env(env_id)    # With EpisodicLifeEnv
eval_env = make_eval_env(env_id)          # Without EpisodicLifeEnv
```

### 3. Inconsistent Episode Counting

**Problem:** Counting life-based episodes during training vs. full episodes during eval.

**Solution:** Track both metrics:
```python
# Track life-based training episodes
training_episodes = 0  # Increments on every done=True

# Track full game episodes separately
full_game_episodes = 0  # Increments only on real game over
if info.get('episode') is not None:  # Gymnasium convention
    full_game_episodes += 1
```

### 4. Not Handling FireReset

**Problem:** Some Atari games (Breakout, etc.) require pressing FIRE to start.

**Solution:** Use FireResetEnv wrapper:
```python
if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
```

### 5. Mixing Up terminated and truncated

**Problem:** Only checking `terminated` and missing time-limit episodes.

**Solution:** Always use combined done flag:
```python
done = result['terminated'] or result['truncated']
if done:
    obs, info = env.reset()
```

---

## Summary

**Key Takeaways:**

1. **Training Mode:**
   - Use `EpisodicLifeEnv` for games with lives (faster learning)
   - Use `NoopResetEnv` for stochastic starts
   - Reset environment on `done = terminated or truncated`
   - Track per-episode metrics (return, length)

2. **Evaluation Mode:**
   - NO `EpisodicLifeEnv` (run full episodes)
   - Use low epsilon (default: 0.05) or greedy (0.0)
   - Disable learning (`model.eval()`, `no_grad()`)
   - Report mean/median/std over multiple episodes (default: 10 episodes)
   - **Implemented in:** `src/training/evaluation.py::evaluate()` function
   - **Configuration:** `eval.epsilon`, `eval.num_episodes` in config files

3. **Episode Tracking:**
   - Accumulate rewards until done
   - Reset counters on episode end
   - Compute rolling statistics for training progress
   - Log both individual and aggregate metrics

4. **Configuration:**
   - Expose `episodic_life`, `noop_max`, `frameskip` in config
   - Create separate train/eval environment builders
   - Document wrapper order and reasoning
   - **Config files:** `experiments/dqn_atari/configs/base.yaml`
     - `training.episode_life`: Enable life-loss as terminal (default: true)
     - `env.max_noop_start`: Random no-ops on reset (default: 30)
     - `env.frameskip`: Action repeat (default: 4)
     - `eval.epsilon`: Evaluation exploration rate (default: 0.05)
     - `eval.num_episodes`: Episodes per evaluation (default: 10)
     - `eval.interval`: Frames between evaluations (default: 250K)

**Implementation Checklist:**
- [ ] Implement `make_training_env()` with all wrappers
- [ ] Implement `make_eval_env()` without life-loss wrapper
- [ ] Create `EpisodeTracker` class for metrics
- [ ] Add proper reset logic after episode end
- [ ] Expose configuration flags for episode handling
- [ ] Test with and without `EpisodicLifeEnv`
- [ ] Verify eval returns are higher than training returns (due to full episodes)

**Related Components (Subtask 6):**

- **Evaluation System:** `src/training/evaluation.py` - Periodic performance assessment with low-ε policy
  - `evaluate()`: Run N episodes with eval_epsilon (default: 0.05, 10 episodes)
  - `EvaluationScheduler`: Trigger evaluations every 250K frames
  - `EvaluationLogger`: Log mean/median/std returns to CSV and JSON

- **Reference-State Q Tracking:** `src/training/q_tracking.py` - Monitor learning via Q-values
  - `ReferenceStateQTracker`: Track Q-values on fixed batch of states
  - Provides smooth learning signal when episode returns are noisy
  - Default: 100 reference states, logged every 10K steps

- **Episode Logging:** `src/training/logging.py` - Structured episode metrics
  - `EpisodeLogger`: Log return, length, rolling statistics
  - Handles both life-based (training) and full episodes (eval)

**Next Steps:**
See `docs/design/training_loop_runtime.md` for complete training loop orchestration that integrates episode handling, evaluation, and Q tracking with the DQN update pipeline.
