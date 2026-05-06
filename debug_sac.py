"""
SAC debug/training script for Pendulum-v1 with full metric tracking and plots.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
from tqdm import trange

from sac_assignment_jax import (
    init_policy,
    init_critic,
    ReplayBuffer,
    policy_sample,
    scale_action,
    critic_update,
    actor_update,
    alpha_update,
    soft_update,
)


def train_and_plot(
    env,
    num_steps: int = 100_000,
    batch_size: int = 256,
    warmup_steps: int = 5000,
    target_entropy: float | None = None,
    hidden_dim: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr: float = 3e-4,
):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if target_entropy is None:
        target_entropy = float(action_dim)

    key = jax.random.PRNGKey(0)

    key, pk, ck, tck = jax.random.split(key, 4)
    policy_params = init_policy(state_dim, action_dim, hidden_dim, key=pk)
    critic_params = init_critic(state_dim, action_dim, hidden_dim, key=ck)
    target_critic_params = jax.tree.map(lambda x: x, critic_params)
    buffer = ReplayBuffer(state_dim, action_dim)

    log_alpha = jnp.array(0.0)
    alpha = jnp.exp(log_alpha)

    critic_optimizer = optax.adam(lr)
    actor_optimizer = optax.adam(lr)
    alpha_optimizer = optax.adam(lr)

    critic_opt_state = critic_optimizer.init(critic_params)
    policy_opt_state = actor_optimizer.init(policy_params)
    alpha_opt_state = alpha_optimizer.init(log_alpha)

    state, _ = env.reset()
    episode_return = 0.0
    recent_returns = deque(maxlen=20)

    # ---- metric accumulators ----
    steps_log = []
    critic_loss_log = []
    actor_loss_log = []
    entropy_log = []
    alpha_log = []
    q1_log = []
    q2_log = []
    episode_step_log = []
    episode_return_log = []

    pbar = trange(num_steps, desc="SAC", unit="step")
    for step in pbar:
        # ---- collect transition ----
        if step < warmup_steps:
            action = env.action_space.sample()
        else:
            key, sample_key = jax.random.split(key)
            action_batch, _ = policy_sample(
                policy_params, jnp.array(state[None]), sample_key
            )
            action = scale_action(np.array(action_batch[0]), env.action_space)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(state, action / 2, reward, next_state, done)

        episode_return += reward
        state = next_state

        if done:
            recent_returns.append(episode_return)
            episode_step_log.append(step)
            episode_return_log.append(episode_return)
            state, _ = env.reset()
            episode_return = 0.0

        # ---- update ----
        if step >= warmup_steps and len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)

            key, ck_ = jax.random.split(key)
            critic_params, critic_opt_state, critic_info = critic_update(
                critic_params, target_critic_params, policy_params,
                critic_opt_state, batch, alpha, gamma, ck_, critic_optimizer,
            )

            key, ak_ = jax.random.split(key)
            policy_params, policy_opt_state, log_probs, actor_info = actor_update(
                policy_params, critic_params,
                policy_opt_state, batch, alpha, ak_, actor_optimizer,
            )

            log_alpha, alpha_opt_state, alpha, alpha_info = alpha_update(
                log_alpha, alpha_opt_state, log_probs, target_entropy, alpha_optimizer,
            )

            target_critic_params = soft_update(critic_params, target_critic_params, tau)

            steps_log.append(step)
            critic_loss_log.append(float(critic_info["critic_loss"]))
            actor_loss_log.append(float(actor_info["actor_loss"]))
            entropy_log.append(float(actor_info["entropy"]))
            alpha_log.append(float(alpha_info["alpha"]))
            q1_log.append(float(critic_info["q1_mean"]))
            q2_log.append(float(critic_info["q2_mean"]))

        # ---- tqdm status ----
        if recent_returns:
            pbar.set_postfix(
                ret=f"{np.mean(recent_returns):.1f}",
                alpha=f"{alpha:.3f}",
            )

    env.close()

    # ---- plotting ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), tight_layout=True)

    axes[0, 0].plot(episode_step_log, episode_return_log, alpha=0.4, linewidth=0.8)
    if len(episode_return_log) >= 20:
        smoothed = np.convolve(episode_return_log, np.ones(20) / 20, mode="valid")
        axes[0, 0].plot(
            episode_step_log[19:], smoothed, color="tab:red", linewidth=1.5
        )
    axes[0, 0].set_title("Episode Return")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("return")

    axes[0, 1].plot(steps_log, critic_loss_log, linewidth=0.5, alpha=0.6)
    axes[0, 1].set_title("Critic Loss")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("loss")

    axes[1, 0].plot(steps_log, actor_loss_log, linewidth=0.5, alpha=0.6)
    axes[1, 0].set_title("Actor Loss")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("loss")

    axes[1, 1].plot(steps_log, entropy_log, linewidth=0.8, label="entropy")
    axes[1, 1].axhline(target_entropy, color="tab:red", linestyle="--", label="target")
    axes[1, 1].set_title("Entropy")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend()

    axes[2, 0].plot(steps_log, alpha_log, linewidth=0.8)
    axes[2, 0].set_title("Alpha (temperature)")
    axes[2, 0].set_xlabel("step")
    axes[2, 0].set_ylabel("α")

    axes[2, 1].plot(steps_log, q1_log, linewidth=0.5, alpha=0.6, label="Q1")
    axes[2, 1].plot(steps_log, q2_log, linewidth=0.5, alpha=0.6, label="Q2")
    axes[2, 1].set_title("Mean Q Values")
    axes[2, 1].set_xlabel("step")
    axes[2, 1].legend()

    fig.savefig("sac_pendulum_diagnostics.png", dpi=150)
    print("Saved plot to sac_pendulum_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    train_and_plot(env, num_steps=100_000)
