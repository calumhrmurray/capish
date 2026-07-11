import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import corner
import matplotlib.pyplot as plt
import numpy as np
import corner, sys
import emcee
import getdist
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import numpy as np

def preprocess_x(x_raw, x_mean, x_std):
    """Apply to any raw x before feeding to the flow."""
    return (np.log1p(x_raw) - x_mean) / x_std

def norm_theta(t, theta_mean, theta_std):     return (t - theta_mean) / theta_std
def unnorm_theta(t_n, theta_mean, theta_std): return t_n * theta_std + theta_mean

def train_sbi(theta_raw, x_raw, N_STEPS = 2000, normalize = True):
    THETA_DIM = theta_raw.shape[1]   # 6
    X_DIM     = x_raw.shape[1]       # 40
    
    print(f"x shape     : {x_raw.shape}")
    print(f"theta shape : {theta_raw.shape}")
    
    
    # ── 2. Train / validation split ───────────────────────────────────────────────
    
    N        = len(x_raw)
    n_val    = int(0.1 * N)
    idx_perm = np.random.default_rng(0).permutation(N)
    
    x_train,     x_val     = x_raw[idx_perm[n_val:]],     x_raw[idx_perm[:n_val]]
    theta_train, theta_val = theta_raw[idx_perm[n_val:]], theta_raw[idx_perm[:n_val]]
    
    
    # ── 3. Normalisation ──────────────────────────────────────────────────────────
    #
    #  x     : log-transform (counts can span orders of magnitude) then standardise
    #          using training set statistics — this is unavoidable for x since we
    #          have no prior on the summary statistics.
    #          At inference time you apply the same transform to x_obs.
    #
    #  theta : prior-based standardisation only (Uniform[a,b]):
    #              mean = (a+b)/2,  std = (b-a)/sqrt(12)
    #          No training data touched → no information leak.
    
    # --- x: log1p then standardise ---
    if normalize:
        x_train_log1p = np.log1p(x_train)
        x_val_log1p   = np.log1p(x_val)
        
        x_mean_log1p = x_train_log1p.mean(0)
        x_std_log1p  = x_train_log1p.std(0) + 1e-8
        
        x_train_norm = preprocess_x(x_train, x_mean_log1p, x_std_log1p)
        x_val_norm   = preprocess_x(x_val, x_mean_log1p, x_std_log1p)
        
        
        # --- theta: prior-based ---
        THETA_MIN = theta_raw.min(0)   # inferred from prior support; replace with known bounds
        THETA_MAX = theta_raw.max(0)   # e.g. np.array([...]) if you know them analytically
        
        theta_mean = (THETA_MIN + THETA_MAX) / 2.0
        theta_std  = (THETA_MAX - THETA_MIN) / np.sqrt(12.0)
        
        theta_train_norm = norm_theta(theta_train, theta_mean, theta_std)
        theta_val_norm   = norm_theta(theta_val, theta_mean, theta_std)
    
        normalisation_setup = {}
        normalisation_setup['x_mean_log1p'] = x_mean_log1p
        normalisation_setup['x_std_log1p'] = x_std_log1p
        normalisation_setup['theta_mean'] = theta_mean
        normalisation_setup['theta_std'] = theta_std
        
        # Convert to jnp
        x_train_norm     = jnp.array(x_train_norm,     dtype=jnp.float32)
        x_val_norm       = jnp.array(x_val_norm,       dtype=jnp.float32)
        theta_train_norm = jnp.array(theta_train_norm, dtype=jnp.float32)
        theta_val_norm   = jnp.array(theta_val_norm,   dtype=jnp.float32)
    else: 
        normalisation_setup = {}
        x_train_norm     = jnp.array(x_train,     dtype=jnp.float32)
        x_val_norm       = jnp.array(x_val,       dtype=jnp.float32)
        theta_train_norm = jnp.array(theta_train, dtype=jnp.float32)
        theta_val_norm   = jnp.array(theta_val,   dtype=jnp.float32)
    
    
    # ── 4. Stacked conditional normalizing flow ───────────────────────────────────
    #
    #  N_LAYERS affine coupling layers applied in sequence.
    #  Each layer has its own conditioner network: x → (shift_d, log_scale_d).
    #  Between layers we permute the theta dimensions so every dimension
    #  gets transformed by multiple layers.
    #
    #  log p(theta | x) = log p(z) + sum_layers log|det J_layer|
    #
    #  This is the core fix for broad contours: more expressive than a single layer.
    
    N_LAYERS = 5   # 6–10 is a good range; more = more expressive, slower to train
    
    
    # ── MAF: one MADE network per layer ──────────────────────────────────────────
    #
    #  Each autoregressive conditioner takes (x, theta_{<d}) and outputs
    #  (shift_d, log_scale_d) for dimension d.
    #  This is implemented as a sequential pass over dimensions using a single MLP
    #  that takes [x || theta_masked] as input, which is equivalent to MADE.
    #
    #  Forward (theta → z):  sequential over d, O(D) — fast
    #  Inverse (z → theta):  sequential over d, O(D) — one pass per dimension
    #  Both are O(1) in n_samples when vmapped.
    
    class AutoregressiveConditioner(eqx.Module):
        """
        For dimension d, maps [x (X_DIM) || theta_{0..d-1} (d zeros for d=0)]
        to (shift_d, log_scale_d).
        Implemented as a single MLP over the full concatenated input;
        masking is handled by zeroing out future dimensions before the call.
        """
        net: eqx.nn.MLP
    
        def __init__(self, key):
            self.net = eqx.nn.MLP(
                in_size    = X_DIM + THETA_DIM,   # x + theta context
                out_size   = THETA_DIM * 2,
                width_size = 256,
                depth      = 4,
                activation = jax.nn.tanh,
                key        = key,
            )
    
        def __call__(self, x, theta_context):
            """
            x             : (X_DIM,)
            theta_context : (THETA_DIM,)  — dimensions >= current d must be zeroed by caller
            Returns shift, log_scale each (THETA_DIM,)
            """
            inp       = jnp.concatenate([x, theta_context])
            out       = self.net(inp)
            shift     = out[:THETA_DIM]
            log_scale = jnp.tanh(out[THETA_DIM:])
            return shift, log_scale
    
    
    class MAFlow(eqx.Module):
        """
        Masked Autoregressive Flow (MAF) with N_LAYERS layers.
        Each layer is one autoregressive pass; layers alternate ordering direction
        to ensure every pair of dimensions interacts within 2 layers.
        """
        conditioners:     list
        orderings:        tuple = eqx.field(static=True)
        inv_orderings:    tuple = eqx.field(static=True)
    
        def __init__(self, key, n_layers=N_LAYERS):
            keys = jax.random.split(key, n_layers)
            self.conditioners = [AutoregressiveConditioner(k) for k in keys]
            # alternate forward / reversed ordering each layer
            fwd = np.arange(THETA_DIM)
            rev = fwd[::-1].copy()
            orders     = [fwd if i % 2 == 0 else rev for i in range(n_layers)]
            inv_orders = [np.argsort(o) for o in orders]
            self.orderings     = tuple(orders)
            self.inv_orderings = tuple(inv_orders)
    
        # ── single MAF layer, forward ─────────────────────────────────────────────
        def _layer_forward(self, theta, x, conditioner, order, inv_order):
            """
            Apply one autoregressive affine layer in the given dimension order.
            Returns z (same shape as theta) and log_det scalar.
            """
            theta_ord = theta[order]          # reorder dims for this layer
    
            def scan_fn(carry, d):
                theta_context = carry         # (THETA_DIM,) — future dims zeroed below
                mask          = jnp.arange(THETA_DIM) < d
                masked_ctx    = theta_ord * mask
                shift, log_scale = conditioner(x, masked_ctx)
                z_d = (theta_ord[d] - shift[d]) * jnp.exp(-log_scale[d])
                # update carry: slot d is now the transformed value
                carry = carry.at[d].set(z_d)
                return carry, (z_d, log_scale[d])
    
            _, (z_ord, log_scales) = jax.lax.scan(
                scan_fn,
                jnp.zeros(THETA_DIM),
                jnp.arange(THETA_DIM),
            )
    
            z = z_ord[inv_order]             # restore original ordering
            log_det = -log_scales.sum()
            return z, log_det
    
        # ── single MAF layer, inverse ─────────────────────────────────────────────
        def _layer_inverse(self, z, x, conditioner, order, inv_order):
            """Inverse of one autoregressive layer (z → theta). Also sequential in d."""
            z_ord = z[order]
    
            def scan_fn(carry, d):
                theta_context = carry
                mask          = jnp.arange(THETA_DIM) < d
                masked_ctx    = theta_context * mask
                shift, log_scale = conditioner(x, masked_ctx)
                theta_d = z_ord[d] * jnp.exp(log_scale[d]) + shift[d]
                carry   = carry.at[d].set(theta_d)
                return carry, theta_d
    
            _, theta_ord = jax.lax.scan(
                scan_fn,
                jnp.zeros(THETA_DIM),
                jnp.arange(THETA_DIM),
            )
    
            return theta_ord[inv_order]
    
        # ── full flow ─────────────────────────────────────────────────────────────
        def log_prob(self, theta, x):
            z = theta
            log_det_total = 0.0
            for conditioner, order, inv_order in zip(
                self.conditioners, self.orderings, self.inv_orderings
            ):
                z, ld = self._layer_forward(z, x, conditioner, order, inv_order)
                log_det_total += ld
            base_lp = -0.5 * (THETA_DIM * jnp.log(2.0 * jnp.pi) + jnp.sum(z ** 2))
            return base_lp + log_det_total
    
        def sample(self, key, x, n_samples):
            z = jax.random.normal(key, (n_samples, THETA_DIM))
    
            def sample_one(z_single):
                theta = z_single
                for conditioner, order, inv_order in zip(
                    reversed(self.conditioners),
                    reversed(self.orderings),
                    reversed(self.inv_orderings),
                ):
                    theta = self._layer_inverse(theta, x, conditioner, order, inv_order)
                return theta
    
            return jax.vmap(sample_one)(z)   # (n_samples, THETA_DIM)
    
    
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    model = MAFlow(subkey)
    
    
    # ── 5. Training ───────────────────────────────────────────────────────────────
    
    def loss_fn(model, theta_batch, x_batch):
        lps = jax.vmap(model.log_prob)(theta_batch, x_batch)
        return -jnp.mean(lps)
    
    LR         = 1e-3
    BATCH_SIZE = 256
    N_STEPS    = N_STEPS   # more steps to compensate for harder optimisation landscape
    
    scheduler = optax.cosine_decay_schedule(LR, N_STEPS)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(scheduler))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    
    @eqx.filter_jit
    def train_step(model, opt_state, theta_batch, x_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, theta_batch, x_batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    N_TRAIN = len(x_train_norm)
    train_losses, val_losses = [], []
    
    print("Training …")
    for step in range(N_STEPS):
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, N_TRAIN, (BATCH_SIZE,), replace=False)
    
        model, opt_state, loss = train_step(
            model, opt_state, theta_train_norm[idx], x_train_norm[idx]
        )
        train_losses.append(float(loss))
    
        if step % 500 == 0 or step == N_STEPS - 1:
            val_loss = float(loss_fn(model, theta_val_norm, x_val_norm))
            val_losses.append((step, val_loss))
            print(f"  step {step:5d}  train={loss:.4f}  val={val_loss:.4f}")
    
    print("Training complete.")

    return model, train_losses, val_losses, normalisation_setup