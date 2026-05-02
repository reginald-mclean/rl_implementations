"""
SAC (Soft Actor-Critic) Implementation Assignment — JAX Version
===============================================================
Implement each component of SAC from scratch using JAX.

Key JAX concepts you'll need:
    - jax.numpy (jnp)  : drop-in numpy replacement, but functional + traceable
    - jax.random       : explicit PRNG keys (no global state)
    - jax.grad         : differentiate any scalar-output function
    - jax.jit          : compile functions for speed
    - jax.tree.map     : apply a function over nested dicts/lists of arrays

Weights are plain dicts — no classes with mutable state.
Every function is pure: same inputs always produce same outputs.

Run individual sections with:
    python sac_assignment_jax.py --section 1   # policy
    python sac_assignment_jax.py --section 2   # critic
    python sac_assignment_jax.py --section 3   # replay buffer
    python sac_assignment_jax.py --section 4   # updates
    python sac_assignment_jax.py --section 5   # sanity checks
    python sac_assignment_jax.py --section all # full training loop
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Optional
import argparse

from collections import deque

# JAX uses explicit PRNG keys. Split a root key to get independent streams.
# e.g.:
#   key = jax.random.PRNGKey(42)
#   key, subkey = jax.random.split(key)
#   samples = jax.random.normal(subkey, shape=(batch, dim))

ROOT_KEY = jax.random.PRNGKey(42)


# ==============================================================================
# PART 1: The Policy Network
# ==============================================================================
# A squashed Gaussian policy. Weights are stored in a plain dict.
# The network takes a state and outputs a mean and log standard deviation.
# Actions are squashed to (-1, 1) via tanh.
#
# Questions to answer before coding:
#   Q1. Why do we clamp log_std rather than std directly?
#   Q2. Why does the tanh correction term exist? What goes wrong without it?
#   Q3. Why must we use the reparameterization trick rather than just
#       jax.random.normal(key, mean, std)?
# ==============================================================================

def init_policy(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Dict:
    """
    Initialize policy weights as a plain dict.
    Architecture:
        state -> Linear(hidden_dim) -> ReLU
              -> Linear(hidden_dim) -> ReLU
              -> Linear(action_dim) [mean head]
              -> Linear(action_dim) [log_std head]

    Use jax.random.normal + * 0.01 for weights, jnp.zeros for biases.
    Split the key for each weight matrix (jax.random.split).

    Returns a dict with keys:
        'w1', 'b1', 'w2', 'b2', 'w_mean', 'b_mean', 'w_log_std', 'b_log_std'
    """
    key, sub1, sub2, sub3, sub4 = jax.random.split(key, 5)
    w1 = jax.random.normal(sub1, shape=(state_dim, hidden_dim)) * 0.01
    b1 = jnp.zeros(hidden_dim)
    w2 = jax.random.normal(sub2, shape=(hidden_dim, hidden_dim)) * 0.01
    b2 = jnp.zeros(hidden_dim)
    w_mean = jax.random.normal(sub3, shape=(hidden_dim, action_dim)) * 0.01
    b_mean = jnp.zeros(action_dim)
    w_log_std = jax.random.normal(sub4, shape=(hidden_dim, action_dim)) * 0.01
    b_log_std = jnp.zeros(action_dim)


    return {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2, 'w_mean': w_mean, 'b_mean': b_mean, 'w_log_std': w_log_std, 'b_log_std': b_log_std}

def policy_forward(
    params: Dict,
    state: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Forward pass through the policy network.

    Args:
        params: dict of weights from init_policy
        state:  (batch, state_dim)

    Returns:
        mean:    (batch, action_dim)
        log_std: (batch, action_dim), clamped to [-20, 2]

    Use jnp.dot, jax.nn.relu, jnp.clip.
    """

    x = jnp.dot(state, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['w2']) + params['b2']
    x = jax.nn.relu(x)
    means = jnp.dot(x, params['w_mean']) + params['b_mean']
    log_std = jnp.dot(x, params['w_log_std']) + params['b_log_std']
    log_std = jnp.clip(log_std, -20, 2)

    return means, log_std


def policy_sample(
    params: Dict,
    state: jax.Array,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Sample an action using the reparameterization trick and compute its
    log probability under the squashed Gaussian.

    Note: key is now an explicit argument — JAX has no global RNG state.
    Use jax.random.normal(key, shape) to sample epsilon.

    Steps:
        1. mean, log_std = policy_forward(params, state)
        2. std = exp(log_std)
        3. eps ~ N(0, I)           via jax.random.normal(key, mean.shape)
        4. pre_tanh = mean + std * eps
        5. action = tanh(pre_tanh)
        6. log_prob = Gaussian log prob on eps - tanh correction
                    = -0.5 * sum(eps**2 + log(2*pi), axis=-1)
                      - sum(log(1 - action**2 + 1e-6), axis=-1)

    Args:
        params: policy weights
        state:  (batch, state_dim)
        key:    JAX PRNGKey

    Returns:
        action:   (batch, action_dim), in (-1, 1)
        log_prob: (batch,)
    """

    mean, log_std = policy_forward(params, state)
    std = jnp.exp(log_std)
    key, eps_key = jax.random.split(key)
    eps = jax.random.normal(eps_key, mean.shape)
    pre_tanh = mean + std * eps
    action = jnp.tanh(pre_tanh)
    log_prob = -0.5 * jnp.sum(eps**2 + jnp.log(2*jnp.pi), axis=-1) - jnp.sum(jnp.log(1-action**2 + 1e-6), axis=-1)
    return action, log_prob




# ==============================================================================
# PART 2: The Critic Network
# ==============================================================================
# Two Q-networks mapping (state, action) -> scalar Q value.
#
# Questions to answer before coding:
#   Q1. Why two critics instead of one?
#   Q2. What failure mode does the min-Q trick prevent?
# ==============================================================================

def init_critic(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Dict:
    """
    Initialize weights for TWO independent critic networks as a single dict.
    Architecture (each):
        concat(state, action) -> Linear(hidden_dim) -> ReLU
                              -> Linear(hidden_dim) -> ReLU
                              -> Linear(1)

    Use jax.random.normal * 0.01 for weights, jnp.zeros for biases.

    Returns a dict with keys for critic 1 and critic 2, e.g.:
        'w1_1', 'b1_1', 'w2_1', 'b2_1', 'w3_1', 'b3_1',
        'w1_2', 'b1_2', 'w2_2', 'b2_2', 'w3_2', 'b3_2'
    """
    key, sub1, sub2, sub3, sub4, sub5, sub6 = jax.random.split(key, 7)
    w1_1 = jax.random.normal(sub1, shape=(state_dim+action_dim, hidden_dim)) * 0.01
    w2_1 = jax.random.normal(sub2, shape=(hidden_dim, hidden_dim)) * 0.01
    w3_1 = jax.random.normal(sub3, shape=(hidden_dim, )) * 0.01
    b1_1 = jnp.zeros(hidden_dim)
    b2_1 = jnp.zeros(hidden_dim)
    b3_1 = jnp.zeros(1)

    w1_2 = jax.random.normal(sub4, shape=(state_dim+action_dim, hidden_dim)) * 0.01
    w2_2 = jax.random.normal(sub5, shape=(hidden_dim, hidden_dim)) * 0.01
    w3_2 = jax.random.normal(sub6, shape=(hidden_dim, )) * 0.01
    b1_2 = jnp.zeros(hidden_dim)
    b2_2 = jnp.zeros(hidden_dim)
    b3_2 = jnp.zeros(1)

    return {'w1_1': w1_1, 'b1_1': b1_1, 'w2_1': w2_1, 'b2_1': b2_1, 'w3_1': w3_1, 'b3_1': b3_1, 'w1_2': w1_2, 'b1_2': b1_2, 'w2_2': w2_2, 'b2_2': b2_2, 
        'w3_2': w3_2, 'b3_2': b3_2}


def critic_forward(
    params: Dict,
    state: jax.Array,
    action: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Forward pass through both critics.

    Args:
        params: dict of weights from init_critic
        state:  (batch, state_dim)
        action: (batch, action_dim)

    Returns:
        q1: (batch,)
        q2: (batch,)

    Use jnp.concatenate, jnp.dot, jax.nn.relu.
    """

    x = jnp.concatenate([state, action], axis=-1)
    q1 = jnp.dot(x, params['w1_1']) + params['b1_1']
    q1 = jax.nn.relu(q1)
    q1 = jnp.dot(q1, params['w2_1']) + params['b2_1']
    q1 = jax.nn.relu(q1)
    q1 = jnp.dot(q1, params['w3_1']) + params['b3_1']

    q2 = jnp.dot(x, params['w1_2']) + params['b1_2']
    q2 = jax.nn.relu(q2)
    q2 = jnp.dot(q2, params['w2_2']) + params['b2_2']
    q2 = jax.nn.relu(q2)
    q2 = jnp.dot(q2, params['w3_2']) + params['b3_2']


    return q1, q2


# ==============================================================================
# PART 3: The Replay Buffer
# ==============================================================================
# Off-policy SAC collects transitions and stores them here.
# This is the one stateful component — we use a simple class backed by
# numpy arrays (not JAX arrays) for efficient random-access indexing.
#
# Questions to answer before coding:
#   Q1. Why store mask = 1 - done rather than done itself?
#   Q2. Why does decoupling data collection from training help?
# ==============================================================================

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6)):
        """
        Circular buffer storing (s, a, r, s', mask) transitions.
        Pre-allocate numpy arrays of shape (max_size, dim) for efficiency.
        Track current size and write pointer separately.
        """

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.n_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.masks = np.zeros((max_size, ), dtype=np.float32)

        self.max_size = max_size
        self.pos = 0
        self.full = False


    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a single transition. Overwrite oldest entry if full (circular).
        Store mask = 1 - done (float).
        """
        self.states[self.pos, :] = state
        self.n_states[self.pos, :] = next_state
        self.rewards[self.pos] = reward
        self.actions[self.pos, :] = action
        self.masks[self.pos] = 1 - float(done)


        self.pos += 1
        if self.pos == self.max_size:
            self.full = True


    def sample(self, batch_size: int) -> Dict[str, jax.Array]:
        """
        Sample a random batch of transitions and return as JAX arrays.

        Returns dict with keys:
            'states':      (batch_size, state_dim)
            'actions':     (batch_size, action_dim)
            'rewards':     (batch_size,)
            'next_states': (batch_size, state_dim)
            'masks':       (batch_size,)   -- 1 - done

        Hint: sample indices with np.random.randint, then convert to JAX arrays
        with jnp.array(). This keeps numpy for indexing (fast) and JAX for compute.
        """
        sample_idx = np.random.randint(0, self.pos if not self.full else self.max_size, batch_size)

        states = self.states[sample_idx, :]
        n_states = self.n_states[sample_idx, :]
        actions = self.actions[sample_idx, :]
        rewards = self.rewards[sample_idx]
        masks = self.masks[sample_idx]

        return {'states': jnp.array(states), 'next_states': jnp.array(n_states), 'actions': jnp.array(actions), 'rewards': jnp.array(rewards), 'masks': jnp.array(masks)}



    def __len__(self) -> int:
        return self.max_size if self.full else self.pos


# ==============================================================================
# PART 4: The Updates
# ==============================================================================
# Core SAC training logic. Each update returns NEW params — no mutation.
#
# The key JAX pattern for all gradient updates is:
#
#   def loss_fn(params, ...):
#       ...
#       return scalar_loss     # must return a scalar
#
#   grad_fn = jax.grad(loss_fn)          # or jax.value_and_grad for loss+grads
#   grads = grad_fn(params, ...)
#   new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
#
# jax.value_and_grad returns (loss_value, grads) in one call — prefer this
# when you need the loss value for logging.
# ==============================================================================

def critic_loss_fn(
    critic_params: Dict,
    target_critic_params: Dict,
    policy_params: Dict,
    batch: Dict[str, jax.Array],
    alpha: float,
    gamma: float,
    key: jax.Array,
) -> jax.Array:
    """
    Compute the critic loss (scalar).

    Steps:
        1. Sample next_actions, next_log_probs from policy at next_states
        2. next_q1, next_q2 = critic_forward(target_critic_params, ...)
           next_q = min(next_q1, next_q2) - alpha * next_log_probs
           target = rewards + gamma * masks * next_q    ← stop gradient here
        3. q1, q2 = critic_forward(critic_params, states, actions)
        4. return MSE(q1, target) + MSE(q2, target)

    *** STOP GRADIENT on target: use jax.lax.stop_gradient(target) ***
    This prevents gradients from flowing through the target network.

    Questions:
        Q1. Why target_critic_params for next Q values rather than critic_params?
        Q2. Why take min of q1 and q2?
        Q3. Why must target be treated as a constant (stop gradient)?
    """

    next_actions, next_log_probs = policy_sample(policy_params, batch['next_states'], key)
    next_q1, next_q2 = critic_forward(target_critic_params, batch['next_states'], next_actions)
    next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_probs
    target = batch['rewards'] + gamma * batch['masks'] * next_q

    target = jax.lax.stop_gradient(target)

    q1, q2 = critic_forward(critic_params, batch['states'], batch['actions'])
    return jnp.mean((q1 - target) ** 2) + jnp.mean((q2 - target) ** 2)



def critic_update(
    critic_params: Dict,
    target_critic_params: Dict,
    policy_params: Dict,
    batch: Dict[str, jax.Array],
    alpha: float,
    gamma: float = 0.99,
    lr: float = 3e-4,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Tuple[Dict, Dict[str, float]]:
    """
    Compute gradients of critic_loss_fn wrt critic_params and apply SGD.

    Use jax.value_and_grad(critic_loss_fn) — differentiate wrt first argument.
    Update with: new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

    Returns:
        new_critic_params: updated critic weights
        info dict with 'critic_loss', 'q1_mean', 'q2_mean'
    """
    loss, grads = jax.value_and_grad(critic_loss_fn)(critic_params, target_critic_params, policy_params, batch, alpha, gamma, key)
    #grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

    new_params = jax.tree.map(lambda p, g: p - lr*g, critic_params, grads)

    q1, q2 = critic_forward(new_params, batch['states'], batch['actions'])

    return new_params, {'critic_loss': loss, 'q1_mean': jnp.mean(q1), 'q2_mean': jnp.mean(q2)}


def actor_loss_fn(
    policy_params: Dict,
    critic_params: Dict,
    batch: Dict[str, jax.Array],
    alpha: float,
    key: jax.Array,
) -> jax.Array:
    actions, log_probs = policy_sample(policy_params, batch['states'], key)
    q1, q2 = critic_forward(critic_params, batch['states'], actions)
    min_q = jnp.minimum(q1, q2)
    return jnp.mean(alpha * log_probs - min_q), log_probs


def actor_update(
    policy_params: Dict,
    critic_params: Dict,
    batch: Dict[str, jax.Array],
    alpha: float,
    lr: float = 3e-4,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Tuple[Dict, jax.Array, Dict[str, float]]:
    """
    Compute gradients of actor_loss_fn wrt policy_params and apply SGD.

    Returns:
        new_policy_params: updated policy weights
        log_probs:         (batch,) — needed for alpha update
        info dict with 'actor_loss', 'entropy'
    """
    (loss, log_probs), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(policy_params, critic_params, batch, alpha, key)
    #grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    new_params = jax.tree.map(lambda p, g: p - lr*g, policy_params, grads)
    return new_params, log_probs, {'actor_loss': loss, 'entropy': -jnp.mean(log_probs)}


'''def alpha_update(
    log_alpha: jax.Array,
    log_probs: jax.Array,
    target_entropy: float,
    lr: float = 3e-4,
) -> Tuple[jax.Array, float, Dict[str, float]]:
    """
    Update the entropy temperature alpha so that policy entropy tracks
    target_entropy.

    Steps:
        1. alpha_loss = -log_alpha * (log_probs + target_entropy).mean()
        2. grad = jax.grad(lambda la: -la * (log_probs + target_entropy).mean())(log_alpha)
        3. new_log_alpha = log_alpha - lr * grad
        4. new_alpha = exp(new_log_alpha)

    Returns:
        new_log_alpha: scalar jax array
        new_alpha:     float
        info dict with 'alpha_loss', 'alpha'

    Questions:
        Q1. Why optimize log_alpha rather than alpha directly?
        Q2. What happens to the loss when current entropy == target_entropy?
        Q3. What does it mean if alpha converges to near zero? Near infinity?
    """

    alpha_loss = -log_alpha * (log_probs + target_entropy).mean()
    grad = jax.grad(lambda la: -la * (log_probs + target_entropy).mean())(log_alpha)
    new_log_alpha = log_alpha + lr * grad
    new_log_alpha = jnp.clip(new_log_alpha, -10, 2)
    new_alpha = jnp.exp(new_log_alpha)
    print(f"deficit={(log_probs + target_entropy).mean():.3f}")
    print(f"  log_alpha={float(log_alpha):.3f}  deficit={(log_probs+target_entropy).mean():.3f}  grad={float(grad):.6f}")

    return new_log_alpha, new_alpha, {'alpha_loss': alpha_loss, 'alpha': new_alpha}'''

def alpha_update(log_alpha, log_probs, target_entropy, lr=3e-4):
    H_current = -log_probs.mean()
    H_target  = jnp.abs(target_entropy)
    # grad of exp(log_alpha) * (H_target - H_current) wrt log_alpha
    # = exp(log_alpha) * (H_target - H_current)
    grad = jnp.exp(log_alpha) * (H_target - H_current)
    new_log_alpha = log_alpha + lr * grad   # increases when H_current < H_target ✓
    new_alpha = float(jnp.exp(new_log_alpha))
    return new_log_alpha, new_alpha, {
        'alpha_loss': float(jnp.exp(log_alpha) * (H_target - H_current)),
        'alpha': new_alpha
    }


def soft_update(
    critic_params: Dict,
    target_critic_params: Dict,
    tau: float = 0.005,
) -> Dict:
    """
    Exponential moving average update of target critic weights.

    For each parameter pair (w, w_target):
        w_target_new = tau * w + (1 - tau) * w_target

    Use jax.tree.map to apply this over all leaves of both dicts.

    Returns new_target_critic_params (do NOT mutate in place).
    """

    new_target_params = jax.tree.map(lambda w, w_t: tau*w + (1-tau)*w_t, critic_params, target_critic_params)

    return new_target_params

# ==============================================================================
# PART 5: Sanity Checks
# ==============================================================================

def run_sanity_checks():
    state_dim, action_dim = 4, 2
    batch_size = 32
    hidden_dim = 64

    key = jax.random.PRNGKey(0)

    print("=" * 60)
    print("Sanity Check 1: Policy")
    print("=" * 60)
    key, init_key, sample_key = jax.random.split(key, 3)
    policy_params = init_policy(state_dim, action_dim, hidden_dim, key=init_key)
    states = jax.random.normal(sample_key, (batch_size, state_dim))

    key, action_key = jax.random.split(key)
    actions, log_probs = policy_sample(policy_params, states, action_key)

    assert actions.shape == (batch_size, action_dim), f"Bad action shape: {actions.shape}"
    assert log_probs.shape == (batch_size,), f"Bad log_prob shape: {log_probs.shape}"
    assert jnp.all(actions > -1) and jnp.all(actions < 1), "Actions outside (-1, 1)"
    assert jnp.all(log_probs < 0), "Log probs should be negative"
    entropy = -log_probs.mean()
    assert entropy > 0, f"Entropy should be positive, got {entropy}"
    print(f"  actions range: [{actions.min():.3f}, {actions.max():.3f}]  (should be in (-1,1))")
    print(f"  mean log_prob: {log_probs.mean():.3f}  (should be negative)")
    print(f"  entropy: {entropy:.3f}  (should be positive)")
    print("  PASSED\n")

    print("=" * 60)
    print("Sanity Check 2: Critic")
    print("=" * 60)
    key, critic_key = jax.random.split(key)
    critic_params = init_critic(state_dim, action_dim, hidden_dim, key=critic_key)
    q1, q2 = critic_forward(critic_params, states, actions)
    assert q1.shape == (batch_size,), f"Bad q1 shape: {q1.shape}"
    assert q2.shape == (batch_size,), f"Bad q2 shape: {q2.shape}"
    print(f"  q1 mean: {q1.mean():.3f}, q2 mean: {q2.mean():.3f}")
    print("  PASSED\n")

    print("=" * 60)
    print("Sanity Check 3: Replay Buffer")
    print("=" * 60)
    buf = ReplayBuffer(state_dim, action_dim, max_size=1000)
    for _ in range(200):
        s  = np.random.randn(state_dim)
        a  = np.random.randn(action_dim)
        r  = float(np.random.randn())
        s2 = np.random.randn(state_dim)
        d  = bool(np.random.rand() < 0.05)
        buf.add(s, a, r, s2, d)
    batch = buf.sample(batch_size)
    assert batch['states'].shape == (batch_size, state_dim)
    assert batch['masks'].shape == (batch_size,), f"{batch['masks'].shape}, {batch_size}"
    assert jnp.all((batch['masks'] == 0) | (batch['masks'] == 1)), "Masks should be 0 or 1"
    print(f"  sampled batch keys: {list(batch.keys())}")
    print("  PASSED\n")

    print("=" * 60)
    print("Sanity Check 4: Alpha Update")
    print("=" * 60)
    log_alpha = jnp.array(0.0)
    target_entropy = float(action_dim)   # +2.0: entropy above this → decrease alpha

    # Case 1: entropy too low (log_probs close to 0) → alpha should increase
    low_entropy_log_probs = jnp.full(batch_size, -0.1)   # entropy=0.1 < target=2.0
    new_log_alpha, new_alpha, _ = alpha_update(log_alpha, low_entropy_log_probs, target_entropy)
    assert new_alpha > float(jnp.exp(log_alpha)), \
        f"Alpha should increase when entropy too low. Got {new_alpha:.4f} vs {float(jnp.exp(log_alpha)):.4f}"
    print(f"  low entropy  -> alpha {float(jnp.exp(log_alpha)):.4f} -> {new_alpha:.4f}  (should increase)")

    # Case 2: entropy too high (log_probs very negative) → alpha should decrease
    high_entropy_log_probs = jnp.full(batch_size, -10.0)  # entropy=10 > target=2.0
    new_log_alpha, new_alpha, _ = alpha_update(log_alpha, high_entropy_log_probs, target_entropy)
    assert new_alpha < float(jnp.exp(log_alpha)), \
        f"Alpha should decrease when entropy too high. Got {new_alpha:.4f} vs {float(jnp.exp(log_alpha)):.4f}"
    print(f"  high entropy -> alpha {float(jnp.exp(log_alpha)):.4f} -> {new_alpha:.4f}  (should decrease)")
    print("  PASSED\n")

    print("=" * 60)
    print("Sanity Check 5: Critic Update (Q moves toward target)")
    print("=" * 60)
    key, target_key = jax.random.split(key)
    target_critic_params = init_critic(state_dim, action_dim, hidden_dim, key=target_key)
    batch = {
        'states':      jnp.array(np.random.randn(batch_size, state_dim)),
        'actions':     jnp.array(np.random.randn(batch_size, action_dim) * 0.1),
        'rewards':     jnp.zeros(batch_size),   # zero reward → Q should go to 0
        'next_states': jnp.array(np.random.randn(batch_size, state_dim)),
        'masks':       jnp.zeros(batch_size),   # terminal → no bootstrap
    }
    alpha = 0.2
    q1_before, q2_before = critic_forward(critic_params, batch['states'], batch['actions'])
    for _ in range(50):
        key, update_key = jax.random.split(key)
        critic_params, info = critic_update(
            critic_params, target_critic_params, policy_params,
            batch, alpha, key=update_key
        )
    q1_after, q2_after = critic_forward(critic_params, batch['states'], batch['actions'])
    print(f"  q1 before: {q1_before.mean():.3f}, after: {q1_after.mean():.3f}  (should move toward 0)")
    print(f"  critic_loss: {info['critic_loss']:.4f}")
    assert abs(float(q1_after.mean())) < abs(float(q1_before.mean())), \
        "Q1 did not move toward 0"
    print("  PASSED\n")

    print("All sanity checks passed!")

def scale_action(action, action_space):
    low, high = action_space.low, action_space.high
    return low + (action + 1.0) * 0.5 * (high - low)


# ==============================================================================
# PART 6: Training Loop
# ==============================================================================

def train(
    env,
    num_steps: int = int(1e6),
    batch_size: int = 256,
    warmup_steps: int = 5000,
    target_entropy: Optional[float] = None,
    hidden_dim: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr: float = 3e-4,
    log_interval: int = 1000,
):
    """
    Full SAC training loop.

    Args:
        env: gymnasium environment with .reset(), .step(),
             .observation_space, .action_space
        num_steps:      total environment steps
        batch_size:     transitions per gradient update
        warmup_steps:   random actions before training starts
        target_entropy: defaults to -action_dim if None

    Order of operations each step:
        1. Collect one transition (random if warming up)
        2. Add to replay buffer
        3. If warmed up: sample batch and run all four updates
        4. Log metrics every log_interval steps

    PRNG discipline:
        - Keep a single `key` variable and split it before every stochastic op:
              key, subkey = jax.random.split(key)
              actions, log_probs = policy_sample(policy_params, state, subkey)
        - Never reuse a key.
    """
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if target_entropy is None:
        target_entropy = float(action_dim)

    key = jax.random.PRNGKey(0)

    # --- Initialize components ---
    key, pk, ck, tck = jax.random.split(key, 4)
    policy_params        = init_policy(state_dim, action_dim, hidden_dim, key=pk)
    critic_params        = init_critic(state_dim, action_dim, hidden_dim, key=ck)
    target_critic_params = init_critic(state_dim, action_dim, hidden_dim, key=tck)
    buffer               = ReplayBuffer(state_dim, action_dim)

    # TODO: copy critic_params into target_critic_params so they start identical
    # Hint: jax.tree.map(lambda x: x, critic_params) returns a copy

    target_critic_params = jax.tree.map(lambda x: x, critic_params)

    log_alpha = jnp.array(0.0)
    alpha     = float(jnp.exp(log_alpha))

    state, _ = env.reset()
    episode_return = 0.0

    episodic_returns = deque(maxlen=20)


    after = sum(jnp.sum(p**2) for p in jax.tree.leaves(policy_params))

    for step in range(num_steps):

        # --- Collect transition ---
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
        buffer.add(state, action/2, reward, next_state, done)

        episode_return += reward
        state = next_state

        if done:
            episodic_returns.append(episode_return)
            state, _ = env.reset()
            episode_return = 0.0

        # --- Update ---
        if step >= warmup_steps and len(buffer) >= batch_size:

            # 2. Check if the policy is actually changing over time
            # Log the mean and std of actions sampled at a fixed state
            test_state = jnp.zeros((1, state_dim))
            key, k = jax.random.split(key)
            test_actions, _ = policy_sample(policy_params, test_state, k)
            #print(f"test action: {float(test_actions[0,0]):.4f}")

            batch = buffer.sample(batch_size)

            # Are states and next_states different?
            #print(f"state: {batch['states'][0]}")
            #print(f"next_state: {batch['next_states'][0]}")
            #print(f"reward: {batch['rewards'][0]}")
            #print(f"mask: {batch['masks'][0]}")

            #print(f"reward range in buffer: {batch['rewards'].min():.2f} to {batch['rewards'].max():.2f}")

            # 1. Critic
            key, ck = jax.random.split(key)
            critic_params, critic_info = critic_update(
                critic_params, target_critic_params, policy_params,
                batch, alpha, gamma, lr, key=ck
            )

            # 2. Actor
            before = sum(jnp.sum(p**2) for p in jax.tree.leaves(policy_params))
            key, ak = jax.random.split(key)
            policy_params, log_probs, actor_info = actor_update(
                policy_params, critic_params, batch, alpha, lr, key=ak
            )
            # after = sum(jnp.sum(p**2) for p in jax.tree.leaves(policy_params))
            # print(f"policy param norm change: {float(after - before):.6f}")

            # 3. Alpha
            log_alpha, alpha, alpha_info = alpha_update(
                log_alpha, log_probs, target_entropy, lr
            )

            # 4. Soft update target
            target_critic_params = soft_update(critic_params, target_critic_params, tau)

            # --- Logging ---
            if step % log_interval == 0:
                print(
                    f"step {step:7d} | "
                    f"critic_loss {critic_info['critic_loss']:.4f} | "
                    f"actor_loss {actor_info['actor_loss']:.4f} | "
                    f"entropy {actor_info['entropy']:.4f} | "
                    f"alpha {alpha_info['alpha']:.4f} | "
                    f"episodic returns {np.mean(np.array(episodic_returns))}  | "
                    f"q1_vals: {critic_info['q1_mean']} q2_vals: {critic_info['q2_mean']}"
                )
                after = sum(jnp.sum(p**2) for p in jax.tree.leaves(policy_params))
                print(f"policy param norm change: {float(after - before):.6f}")

# ==============================================================================
# STRETCH GOALS (implement after the above is working)
# ==============================================================================
#
# 1. jit-compile the update functions
#    - Wrap critic_update, actor_update with jax.jit
#    - First call will be slow (tracing); subsequent calls will be fast
#    - Watch out: jit requires static shapes and no Python side effects
#
# 2. C51 Critic
#    - Discretize return distribution onto fixed support [v_min, v_max]
#    - Replace MSE loss with categorical cross-entropy (jax.nn.softmax + log)
#    - Implement distributional Bellman projection (Bellemare et al. 2017)
#
# 3. Second environment
#    - Verify same code trains on Pendulum-v1, HalfCheetah-v4, or Ant-v4
#
# 4. Diagnostics plot
#    - Track per-step: entropy, alpha, mean Q, episode return
#    - Plot all four over training
#    - Does entropy stabilize near target_entropy?
#    - Does alpha converge or keep moving?
#
# ==============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=str, default="checks",
                        choices=["1", "2", "3", "4", "5", "checks", "all"])
    args = parser.parse_args()

    if args.section == "checks":
        run_sanity_checks()

    elif args.section == "all":
        try:
            import gymnasium as gym
        except ImportError:
            print("Install gymnasium:  pip install gymnasium")
            raise

        env = gym.make("Pendulum-v1")
        train(env, num_steps=100_000, log_interval=200)
