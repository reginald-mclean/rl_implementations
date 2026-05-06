"""
PPO (Proximal Policy Optimization) Implementation Assignment — JAX Version
==========================================================================
Implement each component of PPO from scratch using JAX.

Unlike SAC (off-policy, replay buffer), PPO is ON-POLICY:
    - Collect a fixed rollout of N steps using the CURRENT policy
    - Compute advantages over that rollout (GAE)
    - Run K epochs of minibatch updates over that rollout
    - Discard the data and collect fresh rollouts
    - Repeat

Key JAX concepts you'll need:
    - jax.numpy (jnp)        : drop-in numpy, functional + traceable
    - jax.random             : explicit PRNG keys (no global state)
    - jax.grad / jax.value_and_grad : differentiate scalar-output functions
    - jax.jit                : compile functions for speed
    - jax.tree.map           : map over nested dicts/lists of arrays
    - jax.lax.stop_gradient  : block gradient flow through a subexpression

Run individual sections with:
    python ppo_assignment_jax.py --section 1   # policy (actor-critic)
    python ppo_assignment_jax.py --section 2   # rollout buffer
    python ppo_assignment_jax.py --section 3   # GAE
    python ppo_assignment_jax.py --section 4   # updates
    python ppo_assignment_jax.py --section 5   # sanity checks
    python ppo_assignment_jax.py --section all # full training loop
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, List, Optional
import argparse

ROOT_KEY = jax.random.PRNGKey(42)


# ==============================================================================
# PART 1: The Actor-Critic Network
# ==============================================================================
# PPO uses a SHARED backbone with two heads:
#   - Actor head:  outputs action distribution parameters
#   - Critic head: outputs a scalar state-value estimate V(s)
#
# This is different from SAC, which uses entirely separate networks.
# Sharing early layers can improve sample efficiency — both tasks need
# a good state representation.
#
# For continuous actions: Gaussian policy, mean from network, log_std as a
# standalone learnable parameter (not state-dependent).
# For discrete actions: softmax over logits from a Linear head.
#
# We implement CONTINUOUS here. Discrete is a stretch goal.
#
# Questions to answer before coding:
#   Q1. Why might sharing the backbone hurt training stability?
#       (Hint: think about gradient magnitudes from actor vs critic loss)
#   Q2. Why is log_std a standalone parameter here, not a network output?
#       What assumption does this make about the environment?
#   Q3. SAC squashes actions with tanh. PPO typically does NOT squash.
#       What problem can this cause, and how does OpenAI's implementation
#       handle it in practice?
# ==============================================================================

def init_actor_critic(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Dict:
    """
    Initialize actor-critic weights as a plain dict.

    Architecture:
        Shared backbone:
            state -> Linear(hidden_dim) -> Tanh
                  -> Linear(hidden_dim) -> Tanh

        Actor head:
            backbone_out -> Linear(action_dim)    [mean]
            standalone log_std: jnp.zeros(action_dim)

        Critic head:
            backbone_out -> Linear(1)

    Note: PPO typically uses Tanh activations (not ReLU) and orthogonal
    initialization. For simplicity, use jax.random.normal * 0.01 for weights
    and jnp.zeros for biases, except the final actor layer: init with * 0.01,
    and the final critic layer: init with * 1.0.

    Returns a dict with keys:
        'w1', 'b1', 'w2', 'b2',          <- shared backbone
        'w_actor', 'b_actor',              <- actor mean head
        'log_std',                         <- standalone log_std (shape: action_dim)
        'w_critic', 'b_critic'             <- critic head
    """

    key, sub1, sub2, sub3, sub4 = jax.random.split(key, 5)

    w1 = jax.random.normal(sub1, shape=(state_dim, hidden_dim)) * 0.01
    w2 = jax.random.normal(sub2, shape=(hidden_dim, hidden_dim)) * 0.01
    b1 = jnp.zeros(hidden_dim)
    b2 = jnp.zeros(hidden_dim)

    act_out = jax.random.normal(sub3, shape=(hidden_dim, action_dim)) * 0.01
    act_b = jnp.zeros(action_dim)
    log_std = jnp.zeros(action_dim)

    crit_out = jax.random.normal(sub4, shape=(hidden_dim, 1))
    crit_b = jnp.zeros(1)


    return {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2, 'w_actor': act_out, 'b_actor': act_b, 'log_std': log_std, 'w_critic': crit_out, 'b_critic': crit_b}


def actor_critic_forward(
    params: Dict,
    state: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Forward pass through shared backbone + both heads.

    Args:
        params: dict from init_actor_critic
        state:  (batch, state_dim)

    Returns:
        mean:    (batch, action_dim)   action distribution mean
        log_std: (batch, action_dim)   broadcast from params['log_std']
        value:   (batch,)              state value estimate, squeezed

    Use jnp.tanh for activations (not relu — PPO convention).
    Broadcast log_std: jnp.broadcast_to(params['log_std'], mean.shape)
    Squeeze value: value[..., 0]
    """

    x = jnp.dot(state, params['w1']) + params['b1']
    x = jnp.tanh(x)
    x = jnp.dot(x, params['w2']) + params['b2']
    x = jnp.tanh(x)


    means = jnp.dot(x, params['w_actor']) + params['b_actor']
    log_stds = jnp.broadcast_to(params['log_std'], means.shape)

    value = jnp.dot(x, params['w_critic']) + params['b_critic']

    value = jnp.squeeze(value, axis=-1)
    return means, log_stds, value

def sample_action(
    params: Dict,
    state: jax.Array,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sample an action from the Gaussian policy (NO tanh squashing).

    Steps:
        1. mean, log_std, value = actor_critic_forward(params, state)
        2. std = exp(log_std)
        3. eps ~ N(0, I)  via jax.random.normal(key, mean.shape)
        4. action = mean + std * eps    (reparameterization)
        5. log_prob = Gaussian log prob (NO tanh correction)
                    = -0.5 * sum(((action - mean) / std)**2
                                  + 2*log_std
                                  + log(2*pi), axis=-1)

    Args:
        params: actor-critic weights
        state:  (batch, state_dim)
        key:    JAX PRNGKey

    Returns:
        action:   (batch, action_dim)
        log_prob: (batch,)
        value:    (batch,)   critic estimate, used for GAE later

    Note: during rollout collection we detach from the computation graph
    (by converting to numpy). The log_probs returned here are the
    "old" log probs that PPO uses as the denominator of the ratio.
    """
    mean, log_std, value = actor_critic_forward(params, state)
    std = jnp.exp(log_std)

    eps = jax.random.normal(key, mean.shape)
    action = mean + std * eps
    log_prob = -0.5 * jnp.sum(((action - mean) / std) ** 2 + 2*log_std + jnp.log(2 * jnp.pi), axis=-1)
    return action, log_prob, value


def evaluate_actions(
    params: Dict,
    state: jax.Array,
    action: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Re-evaluate stored actions under the CURRENT policy (used during PPO update).
    This is different from sample_action — we're not sampling, we're computing
    the density of ALREADY-SAMPLED actions under the current parameters.

    Steps:
        1. mean, log_std, value = actor_critic_forward(params, state)
        2. log_prob of `action` under N(mean, exp(log_std)):
               log_prob = -0.5 * sum(((action - mean) / std)**2
                                      + 2*log_std
                                      + log(2*pi), axis=-1)
        3. entropy of the Gaussian:
               entropy = 0.5 * sum(log(2*pi*e) + 2*log_std, axis=-1)
                       = 0.5 * action_dim * (1 + log(2*pi)) + log_std.sum(axis=-1)

    Args:
        params: actor-critic weights
        state:  (batch, state_dim)
        action: (batch, action_dim)  — the stored actions from the rollout

    Returns:
        log_prob: (batch,)
        entropy:  (batch,)
        value:    (batch,)

    Questions:
        Q1. Why do we need evaluate_actions instead of reusing sample_action?
        Q2. The entropy term will appear in the loss. What does maximizing
            entropy do to the policy? Why is this useful in PPO?
    """

    mean, log_std, value = actor_critic_forward(params, state)
    std = jnp.exp(log_std)
    log_prob = -0.5 * jnp.sum(((action - mean) / std)**2 + 2 * log_std + jnp.log(2*jnp.pi), axis=-1)
    entropy = 0.5 * action.shape[-1] * (1 + jnp.log(2*jnp.pi)) + log_std.sum(axis=-1)
    return log_prob, entropy, value

# ==============================================================================
# PART 2: The Rollout Buffer
# ==============================================================================
# PPO is ON-POLICY: we collect a fixed-length rollout, compute advantages,
# then iterate over it for K epochs. After K epochs the data is THROWN AWAY.
#
# Unlike the SAC replay buffer (random access, circular, huge), the rollout
# buffer is small, sequential, and cleared after each policy update.
#
# Questions to answer before coding:
#   Q1. Why can't we reuse old data the way SAC does?
#   Q2. The buffer stores log_probs at collection time. Why?
#       (Think about what the PPO ratio computes.)
#   Q3. Why store values in the buffer rather than recomputing them?
# ==============================================================================

class RolloutBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, gamma: float = 0.99, lam: float = 0.95):
        """
        Fixed-size buffer for one rollout.
        Pre-allocate numpy arrays of shape (buffer_size, dim).
        Also store gamma and lam for GAE computation later.

        Fields to pre-allocate:
            states:     (buffer_size, state_dim)
            actions:    (buffer_size, action_dim)
            rewards:    (buffer_size,)
            values:     (buffer_size,)   critic estimates at collection time
            log_probs:  (buffer_size,)   old log probs, for PPO ratio
            dones:      (buffer_size,)   episode termination flags

        Also track:
            pos:  current write index
            full: bool
        """
        self.gamma = gamma
        self.lam = lam
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False


        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)


    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Add a single transition. Advance pos. Mark full if pos == buffer_size.

        Note: unlike SAC's circular buffer, we do NOT wrap around.
        The buffer is reset after each update.
        """
        self.states[self.pos, :] = state
        self.actions[self.pos, :] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done

        self.pos += 1
        self.full = self.pos == self.buffer_size

    def reset(self) -> None:
        """
        Reset pos to 0 and full to False.
        Called after each round of PPO updates.
        """
        self.pos = 0
        self.full = False

    def get(self) -> Dict[str, np.ndarray]:
        """
        Return all stored transitions as a dict of numpy arrays
        (advantages and returns are NOT computed here — see compute_gae).

        Returns dict with keys:
            'states', 'actions', 'rewards', 'values', 'log_probs', 'dones'
        """
        return {'states': self.states, 'actions': self.actions, 'rewards': self.rewards, 'values': self.values, 'log_probs': self.log_probs, 'dones': self.dones}

    def is_full(self) -> bool:
        return self.full


# ==============================================================================
# PART 3: Generalized Advantage Estimation (GAE)
# ==============================================================================
# GAE is the core of PPO's credit assignment. It computes a weighted blend
# of n-step returns, controlled by lambda (lam).
#
# lam = 0 → TD(0): low variance, high bias
# lam = 1 → Monte Carlo: high variance, low bias
# Typical: lam = 0.95
#
# The GAE formula (Schulman et al. 2016):
#
#   delta_t  = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
#   A_t      = delta_t + (gamma * lam) * (1 - done_t) * A_{t+1}
#
# Compute BACKWARDS from t=T-1 to t=0.
#
# Questions to answer before coding:
#   Q1. Why does done_t appear in the recurrence? What goes wrong if you omit it?
#   Q2. Why do we normalize advantages before the update?
#       What could go wrong without normalization?
#   Q3. What is the "return" used for the critic target?
#       How does it relate to the advantage?
# ==============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and returns (critic targets) for a rollout.

    Args:
        rewards:    (T,)   rewards at each step
        values:     (T,)   critic value estimates V(s_t)
        dones:      (T,)   1.0 if episode ended after step t, else 0.0
        last_value: float  V(s_{T+1}), the bootstrap value after the rollout
                           (0.0 if the rollout ended in a terminal state)
        gamma:      discount factor
        lam:        GAE lambda

    Returns:
        advantages: (T,)   GAE estimates, NOT normalized yet
        returns:    (T,)   advantages + values  (used as critic targets)

    Algorithm (iterate BACKWARDS from t = T-1 to 0):
        next_value     = last_value
        next_advantage = 0.0
        for t in reversed(range(T)):
            mask         = 1.0 - dones[t]
            delta        = rewards[t] + gamma * next_value * mask - values[t]
            advantage[t] = delta + gamma * lam * mask * next_advantage
            next_advantage = advantage[t]
            next_value     = values[t]

    Use numpy (not JAX) here — this runs once per rollout, not per minibatch.
    """

    advantages = np.zeros_like(rewards)

    next_value = last_value
    next_adv = 0.0
    for t in reversed(range(len(rewards)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        advantages[t] = delta + gamma * lam * mask * next_advantage
        next_advantage = advantage[t]
        next_value = values[t]

    return advantages, advantages + values




# ==============================================================================
# PART 4: The PPO Update
# ==============================================================================
# The full PPO loss has three terms:
#
#   L = L_clip - c1 * L_value + c2 * L_entropy
#
# where:
#   L_clip:    clipped surrogate objective (actor)
#   L_value:   MSE between critic output and returns
#   L_entropy: mean entropy of the policy (bonus for exploration)
#
# The key PPO innovation is L_clip: instead of TRPO's hard KL constraint,
# we simply clip the probability ratio r_t = pi_new / pi_old to [1-eps, 1+eps].
#
# Questions to answer before coding:
#   Q1. What is the probability ratio r_t = pi_new(a|s) / pi_old(a|s)?
#       Why do we compute it in log-space?
#   Q2. Why do we clip r_t instead of penalizing large KL directly?
#       What does the clip prevent?
#   Q3. Why is L_value included in the same loss as L_clip?
#       What are the tradeoffs of a shared vs separate optimizer?
#   Q4. Why does the entropy bonus help? What happens without it?
# ==============================================================================

def ppo_loss_fn(
    params: Dict,
    states: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[jax.Array, Dict]:
    """
    Compute the full PPO loss (scalar) and diagnostic info.

    Steps:
        1. log_probs, entropy, values = evaluate_actions(params, states, actions)

        2. Normalize advantages (over the minibatch):
               adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
           Use jax.lax.stop_gradient(adv) — advantages are fixed targets.

        3. Probability ratio (log-space for numerical stability):
               log_ratio = log_probs - old_log_probs
               ratio     = exp(log_ratio)

        4. Clipped surrogate objective (MAXIMIZE this — negate for gradient descent):
               surr1 = ratio * adv
               surr2 = clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
               L_clip = mean(min(surr1, surr2))

        5. Value loss (MSE — MINIMIZE this):
               L_value = mean((values - returns)**2)
           Use jax.lax.stop_gradient(returns) — returns are fixed targets.

        6. Entropy bonus (MAXIMIZE this — negate for gradient descent):
               L_entropy = mean(entropy)

        7. Total loss (note signs — we're doing gradient DESCENT):
               loss = -L_clip + value_coef * L_value - entropy_coef * L_entropy

    Args:
        params:        actor-critic weights
        states:        (batch, state_dim)
        actions:       (batch, action_dim)
        old_log_probs: (batch,)   log probs from COLLECTION time
        advantages:    (batch,)   GAE advantages (not yet normalized)
        returns:       (batch,)   GAE returns = advantages + values
        clip_eps:      PPO clip parameter (default 0.2)
        value_coef:    critic loss weight (default 0.5)
        entropy_coef:  entropy bonus weight (default 0.01)

    Returns:
        loss:  scalar
        info:  dict with 'policy_loss', 'value_loss', 'entropy',
                         'approx_kl', 'clip_fraction'

    Extra diagnostics to compute:
        approx_kl    = mean(-log_ratio)        ← monitors policy change
        clip_fraction = mean(|ratio - 1| > clip_eps)  ← fraction of clipped steps
    """
    raise NotImplementedError


def ppo_update(
    params: Dict,
    states: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    lr: float = 3e-4,
) -> Tuple[Dict, Dict]:
    """
    One gradient step on ppo_loss_fn.

    Use jax.value_and_grad(ppo_loss_fn) — differentiate wrt first argument (params).
    Update: new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

    Returns:
        new_params: updated actor-critic weights
        info:       diagnostic dict from ppo_loss_fn
    """
    raise NotImplementedError


def ppo_epoch(
    params: Dict,
    rollout: Dict[str, np.ndarray],
    advantages: np.ndarray,
    returns: np.ndarray,
    n_epochs: int = 10,
    minibatch_size: int = 64,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    lr: float = 3e-4,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Tuple[Dict, Dict]:
    """
    Run n_epochs of minibatch PPO updates over the rollout.

    For each epoch:
        1. Shuffle the rollout indices (use jax.random.permutation(key, T))
        2. Split into minibatches of size minibatch_size
        3. For each minibatch: call ppo_update

    Args:
        params:         actor-critic weights
        rollout:        dict from RolloutBuffer.get()
        advantages:     (T,)  GAE advantages
        returns:        (T,)  GAE returns
        n_epochs:       number of passes over the data
        minibatch_size: transitions per gradient step

    Returns:
        new_params: updated weights after all epochs
        info:       mean of diagnostic values across all updates

    Hint: accumulate info dicts and average at the end.
    jnp.array(list_of_values).mean() works for this.

    Questions:
        Q1. n_epochs > 1 means we update on the same data multiple times.
            Doesn't that violate the on-policy assumption?
            Why does PPO still work?
        Q2. What does approx_kl tell you about whether to stop early?
            (See the "early stopping" stretch goal.)
    """
    raise NotImplementedError


# ==============================================================================
# PART 5: Sanity Checks
# ==============================================================================

def run_sanity_checks():
    state_dim, action_dim = 4, 2
    batch_size = 32
    hidden_dim = 64
    T = 128  # rollout length

    key = jax.random.PRNGKey(0)

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sanity Check 1: Actor-Critic Forward + Sample")
    print("=" * 60)
    key, init_key, state_key, action_key = jax.random.split(key, 4)
    params = init_actor_critic(state_dim, action_dim, hidden_dim, key=init_key)
    states  = jax.random.normal(state_key, (batch_size, state_dim))

    mean, log_std, values = actor_critic_forward(params, states)
    assert mean.shape    == (batch_size, action_dim), f"Bad mean shape: {mean.shape}"
    assert log_std.shape == (batch_size, action_dim), f"Bad log_std shape: {log_std.shape}"
    assert values.shape  == (batch_size,),            f"Bad value shape: {values.shape}"

    actions, log_probs, vals = sample_action(params, states, action_key)
    assert actions.shape   == (batch_size, action_dim), f"Bad action shape: {actions.shape}"
    assert log_probs.shape == (batch_size,),            f"Bad log_prob shape: {log_probs.shape}"
    assert vals.shape      == (batch_size,),            f"Bad value shape: {vals.shape}"
    assert jnp.all(jnp.isfinite(log_probs)),            "log_probs contain NaN/Inf"

    print(f"  mean range:    [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  value range:   [{values.min():.3f}, {values.max():.3f}]")
    print(f"  mean log_prob: {log_probs.mean():.3f}")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sanity Check 2: evaluate_actions consistency")
    print("=" * 60)
    # Actions sampled from the SAME params should give back the same log_probs
    log_probs2, entropy, vals2 = evaluate_actions(params, states, actions)
    assert log_probs2.shape == (batch_size,), f"Bad log_prob shape: {log_probs2.shape}"
    assert entropy.shape    == (batch_size,), f"Bad entropy shape: {entropy.shape}"
    assert jnp.allclose(log_probs, log_probs2, atol=1e-5), \
        f"evaluate_actions disagrees with sample_action log_probs! max diff: {jnp.abs(log_probs - log_probs2).max():.2e}"
    assert jnp.all(entropy > 0), "Entropy should be positive"
    print(f"  max log_prob discrepancy: {jnp.abs(log_probs - log_probs2).max():.2e}  (should be ~0)")
    print(f"  mean entropy: {entropy.mean():.3f}  (should be positive)")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sanity Check 3: RolloutBuffer")
    print("=" * 60)
    buf = RolloutBuffer(state_dim, action_dim, buffer_size=T)
    for i in range(T):
        s   = np.random.randn(state_dim)
        a   = np.random.randn(action_dim)
        r   = float(np.random.randn())
        v   = float(np.random.randn())
        lp  = float(np.random.randn())
        d   = bool(np.random.rand() < 0.05)
        buf.add(s, a, r, v, lp, d)
    assert buf.is_full(), "Buffer should be full after T additions"
    rollout = buf.get()
    assert rollout['states'].shape   == (T, state_dim),  f"Bad states shape: {rollout['states'].shape}"
    assert rollout['actions'].shape  == (T, action_dim), f"Bad actions shape: {rollout['actions'].shape}"
    assert rollout['rewards'].shape  == (T,),            f"Bad rewards shape: {rollout['rewards'].shape}"
    buf.reset()
    assert not buf.is_full(), "Buffer should be empty after reset"
    print(f"  rollout keys: {list(rollout.keys())}")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sanity Check 4: GAE")
    print("=" * 60)
    rewards    = np.ones(T, dtype=np.float32)
    values_np  = np.zeros(T, dtype=np.float32)
    dones      = np.zeros(T, dtype=np.float32)

    # No discounting (gamma=1), no bootstrap, no done flags, all-ones rewards:
    # A_t = 1 + 1 + ... + 1 (T - t terms) — a simple sum
    advantages, returns = compute_gae(rewards, values_np, dones, last_value=0.0, gamma=1.0, lam=1.0)
    assert advantages.shape == (T,), f"Bad advantage shape: {advantages.shape}"
    expected_adv_0 = float(T)     # sum of T ones at t=0
    expected_adv_last = 1.0       # just 1 at t=T-1
    assert abs(advantages[0] - expected_adv_0)    < 1e-3, \
        f"A[0] should be {expected_adv_0:.1f}, got {advantages[0]:.4f}"
    assert abs(advantages[-1] - expected_adv_last) < 1e-3, \
        f"A[T-1] should be {expected_adv_last:.1f}, got {advantages[-1]:.4f}"

    # Returns = advantages + values, so returns == advantages here (values=0)
    assert np.allclose(returns, advantages, atol=1e-5), \
        "returns should equal advantages + values"

    # With done=1 mid-rollout, advantage should reset
    dones_with_cut          = np.zeros(T, dtype=np.float32)
    dones_with_cut[T//2 - 1] = 1.0
    adv_cut, _ = compute_gae(rewards, values_np, dones_with_cut, last_value=0.0, gamma=1.0, lam=1.0)
    assert abs(adv_cut[0] - T//2) < 1e-3, \
        f"With done at T//2, A[0] should be {T//2}, got {adv_cut[0]:.4f}"

    print(f"  A[0]={advantages[0]:.1f} (exp {float(T):.1f}), A[-1]={advantages[-1]:.1f} (exp 1.0)")
    print(f"  A[0] with mid-rollout done: {adv_cut[0]:.1f}  (exp {T//2:.1f})")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sanity Check 5: PPO Loss — ratio = 1 at first update")
    print("=" * 60)
    # If we evaluate actions under the SAME policy that collected them,
    # ratio = pi_new / pi_old = 1, so log_ratio = 0.
    # In that case, L_clip = mean(1 * adv) = mean(adv) = 0 (after normalization)
    # unless we're checking clipping behavior.
    adv_batch  = jnp.ones(batch_size)
    ret_batch  = jnp.ones(batch_size)
    key, state_key, action_key = jax.random.split(key, 3)
    states_b  = jax.random.normal(state_key, (batch_size, state_dim))
    actions_b, old_lps, _ = sample_action(params, states_b, action_key)

    loss, info = ppo_loss_fn(params, states_b, actions_b, old_lps, adv_batch, ret_batch)
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert abs(float(info['approx_kl'])) < 1e-4, \
        f"approx_kl should be ~0 when old==new params, got {info['approx_kl']:.6f}"
    assert abs(float(info['clip_fraction'])) < 1e-4, \
        f"clip_fraction should be ~0 at init, got {info['clip_fraction']:.6f}"
    print(f"  approx_kl:     {float(info['approx_kl']):.6f}  (should be ~0)")
    print(f"  clip_fraction: {float(info['clip_fraction']):.6f}  (should be ~0)")
    print(f"  loss:          {float(loss):.4f}")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sanity Check 6: Critic moves toward returns")
    print("=" * 60)
    # With constant returns = 1.0 and initial values near 0, the critic
    # should move toward 1.0 after several gradient steps.
    target_return = 1.0
    ret_const     = jnp.full(batch_size, target_return)
    adv_zeros     = jnp.zeros(batch_size)
    key, state_key, action_key = jax.random.split(key, 3)
    states_b   = jax.random.normal(state_key, (batch_size, state_dim))
    actions_b, old_lps, _ = sample_action(params, states_b, action_key)

    _, _, v_before = evaluate_actions(params, states_b, actions_b)
    for _ in range(100):
        params, info = ppo_update(
            params, states_b, actions_b, old_lps, adv_zeros, ret_const,
            value_coef=1.0, entropy_coef=0.0, lr=1e-2
        )
    _, _, v_after = evaluate_actions(params, states_b, actions_b)
    print(f"  value before: {v_before.mean():.4f}, after: {v_after.mean():.4f}  (target: {target_return})")
    assert abs(float(v_after.mean()) - target_return) < abs(float(v_before.mean()) - target_return), \
        "Critic did not move toward the target return"
    print("  PASSED\n")

    print("All sanity checks passed!")


# ==============================================================================
# PART 6: Training Loop
# ==============================================================================

def train(
    env,
    total_timesteps: int = int(1e6),
    rollout_steps: int = 2048,       # N: steps collected per policy update
    n_epochs: int = 10,              # K: passes over each rollout
    minibatch_size: int = 64,
    hidden_dim: int = 64,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    log_interval: int = 1,           # log every N policy updates
):
    """
    Full PPO training loop (single environment, synchronous).

    Outer loop (policy updates):
        While timesteps < total_timesteps:
            1. Collect rollout_steps transitions into RolloutBuffer
            2. compute_gae on the completed rollout
            3. ppo_epoch: n_epochs of minibatch updates
            4. RolloutBuffer.reset()
            5. Log metrics

    Rollout collection:
        - step < rollout_steps:
              action, log_prob, value = sample_action(params, state, key)
              next_state, reward, terminated, truncated, _ = env.step(action)
              done = terminated or truncated
              buffer.add(state, action, reward, value, log_prob, done)
              if done: state, _ = env.reset()

    Bootstrap value at rollout end:
        - If last step was NOT terminal:
              _, _, last_value = sample_action(params, last_state, key)
              last_value = float(last_value[0])
        - If terminal:
              last_value = 0.0

    PRNG discipline (same as SAC — split before every stochastic op):
        key, subkey = jax.random.split(key)
        action, log_prob, value = sample_action(params, state[None], subkey)

    Args:
        env: gymnasium environment with .reset(), .step(),
             .observation_space, .action_space

    Differences from SAC loop to notice:
        - No warmup period (PPO doesn't need one — it starts updating immediately)
        - No replay buffer (data is discarded after ppo_epoch)
        - No target network (PPO uses the same network for actor and critic)
        - No alpha/entropy tuning (entropy_coef is fixed)
    """
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    key = jax.random.PRNGKey(0)

    key, init_key = jax.random.split(key)
    params = init_actor_critic(state_dim, action_dim, hidden_dim, key=init_key)
    buffer = RolloutBuffer(state_dim, action_dim, rollout_steps, gamma, lam)

    state, _ = env.reset()
    timesteps = 0
    update_num = 0
    episode_returns = []
    episode_return = 0.0

    while timesteps < total_timesteps:

        # --- Collect rollout ---
        buffer.reset()
        last_done = False

        for _ in range(rollout_steps):
            key, sample_key = jax.random.split(key)
            action_batch, lp_batch, val_batch = sample_action(
                params, jnp.array(state[None]), sample_key
            )
            action   = np.array(action_batch[0])
            log_prob = float(lp_batch[0])
            value    = float(val_batch[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, value, log_prob, done)

            episode_return += reward
            timesteps      += 1
            state           = next_state

            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0
                state, _ = env.reset()

            last_done = done

        # --- Bootstrap ---
        if last_done:
            last_value = 0.0
        else:
            key, v_key = jax.random.split(key)
            _, _, last_val_batch = sample_action(params, jnp.array(state[None]), v_key)
            last_value = float(last_val_batch[0])

        # --- GAE ---
        rollout    = buffer.get()
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'],
            last_value, gamma, lam
        )

        # --- PPO epochs ---
        key, epoch_key = jax.random.split(key)
        params, info = ppo_epoch(
            params, rollout, advantages, returns,
            n_epochs, minibatch_size, clip_eps, value_coef, entropy_coef, lr,
            key=epoch_key
        )

        update_num += 1

        if update_num % log_interval == 0:
            mean_return = np.mean(episode_returns[-10:]) if episode_returns else float('nan')
            print(
                f"update {update_num:5d} | "
                f"timesteps {timesteps:8d} | "
                f"mean_return(10ep) {mean_return:8.2f} | "
                f"policy_loss {info['policy_loss']:.4f} | "
                f"value_loss {info['value_loss']:.4f} | "
                f"entropy {info['entropy']:.4f} | "
                f"approx_kl {info['approx_kl']:.4f} | "
                f"clip_frac {info['clip_fraction']:.4f}"
            )


# ==============================================================================
# STRETCH GOALS (implement after the above is working)
# ==============================================================================
#
# 1. jit-compile the update
#    - Wrap ppo_update with jax.jit
#    - Rollout collection stays in Python/numpy; the GPU-heavy gradient step gets compiled
#    - Tip: make sure arrays passed to jit have consistent dtypes (float32 everywhere)
#
# 2. Early stopping on KL
#    - In ppo_epoch, track approx_kl after each minibatch
#    - If approx_kl exceeds a threshold (e.g. 0.015), break out of the epoch loop
#    - This is what many production PPO implementations do
#
# 3. Gradient clipping
#    - After computing grads, clip global gradient norm before applying:
#          total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
#          scale = jnp.minimum(1.0, max_grad_norm / (total_norm + 1e-8))
#          grads = jax.tree.map(lambda g: g * scale, grads)
#    - Typical max_grad_norm = 0.5
#
# 4. Linear learning rate annealing
#    - Decay lr from initial to 0 over total_timesteps
#    - frac = 1.0 - timesteps / total_timesteps
#    - lr_now = lr * frac
#
# 5. Clipped value loss
#    - PPO paper also clips the value function loss:
#          v_pred_clipped = old_value + clip(value - old_value, -clip_eps, clip_eps)
#          v_loss = max((value - returns)**2, (v_pred_clipped - returns)**2)
#    - This prevents large value updates from destabilizing the shared backbone
#    - Store old_values in the rollout buffer to enable this
#
# 6. Vectorized environments (parallel collection)
#    - Replace single env with gym.vector.make("...", num_envs=N)
#    - Collect N parallel rollouts simultaneously
#    - Greatly increases throughput on CPU
#
# 7. Discrete action spaces
#    - Replace Gaussian policy with Categorical (softmax over logits)
#    - Action sampling: jax.random.categorical(key, logits)
#    - Log prob: log(softmax(logits))[action_index]
#    - Entropy: -sum(softmax * log_softmax, axis=-1)
#    - Test on LunarLander-v2 (discrete) vs LunarLander-v2 (continuous)
#
# 8. Diagnostics
#    - Track per-update: entropy, approx_kl, clip_fraction, value_loss, policy_loss
#    - Track per-episode: return, length
#    - Plot all over training. Key questions:
#        * Does entropy decrease monotonically or stabilize?
#        * Does clip_fraction stay below ~0.1?
#        * Does approx_kl stay well below your early-stopping threshold?
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
        train(env, total_timesteps=500_000, log_interval=5)
