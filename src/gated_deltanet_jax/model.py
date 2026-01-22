
import math
from typing import Optional, Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from .ops import chunk_gated_delta_rule_fwd, fused_recurrent_gated_delta_rule_fwd
from .configuration import GatedDeltaNetConfig

# -----------------------------------------------------------------------------
# Basic Modules
# -----------------------------------------------------------------------------

class ShortConvolution(nnx.Module):
    """
    Short convolution layer using depthwise convolution.
    """
    def __init__(self, hidden_size: int, kernel_size: int, activation='silu', use_bias=False, rngs: nnx.Rngs = None):
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        self.conv = nnx.Conv(
            in_features=hidden_size,
            out_features=hidden_size,
            kernel_size=(kernel_size,),
            feature_group_count=hidden_size,
            use_bias=use_bias,
            padding=[(kernel_size - 1, 0)],
            rngs=rngs,
        )

    def __call__(self, x):
        y = self.conv(x)
        if self.activation == 'silu':
            y = nnx.silu(y)
        elif self.activation == 'swish':
            y = nnx.swish(y)
        return y

class FusedRMSNormGated(nnx.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, rngs: nnx.Rngs = None):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((hidden_size,)))
        
    def __call__(self, x, gate=None):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(var + self.eps)
        y = normed * self.weight
        if gate is not None:
            y = y * nnx.silu(gate)
        return y

# -----------------------------------------------------------------------------
# Attention Layer (GatedDeltaNet)
# -----------------------------------------------------------------------------

class GatedDeltaNet(nnx.Module):
    def __init__(
        self,
        config: GatedDeltaNetConfig,
        layer_idx: int = None,
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_v_heads = config.num_v_heads if config.num_v_heads is not None else config.num_heads
        
        self.head_k_dim = config.head_dim
        self.head_v_dim = int(config.head_dim * config.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        
        self.q_proj = nnx.Linear(config.hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(config.hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_size, self.value_dim, use_bias=False, rngs=rngs)
        
        self.a_proj = nnx.Linear(config.hidden_size, self.num_v_heads, use_bias=False, rngs=rngs)
        self.b_proj = nnx.Linear(config.hidden_size, self.num_v_heads, use_bias=False, rngs=rngs)
        
        if config.use_gate:
            self.g_proj = nnx.Linear(config.hidden_size, self.value_dim, use_bias=False, rngs=rngs)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, rngs=rngs)
        else:
            self.g_proj = None
            self.o_norm = nnx.RMSNorm(self.head_v_dim, rngs=rngs)
            
        self.o_proj = nnx.Linear(self.value_dim, config.hidden_size, use_bias=False, rngs=rngs)
        
        if config.use_short_conv:
            self.q_conv = ShortConvolution(self.key_dim, config.conv_size, use_bias=False, rngs=rngs)
            self.k_conv = ShortConvolution(self.key_dim, config.conv_size, use_bias=False, rngs=rngs)
            self.v_conv = ShortConvolution(self.value_dim, config.conv_size, use_bias=False, rngs=rngs)
        
        # Init A_log and dt_bias
        key1, key2 = jax.random.split(rngs.params(), 2)
        A_init = jax.random.uniform(key1, (self.num_v_heads,), minval=1e-4, maxval=16)
        self.A_log = nnx.Param(jnp.log(A_init))
        
        dt_min, dt_max = 0.001, 0.1
        dt_init = jnp.exp(
            jax.random.uniform(key2, (self.num_v_heads,), minval=math.log(dt_min), maxval=math.log(dt_max))
        )
        dt_init = jnp.maximum(dt_init, 1e-4)
        inv_dt = dt_init + jnp.log(-jnp.expm1(-dt_init))
        self.dt_bias = nnx.Param(inv_dt)

    def __call__(self, x, mask=None):
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if self.config.use_short_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)
        else:
            q = nnx.silu(q)
            k = nnx.silu(k)
            v = nnx.silu(v)
            
        q = q.reshape(B, T, self.num_heads, self.head_k_dim)
        k = k.reshape(B, T, self.num_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
        
        if self.num_v_heads > self.num_heads:
            rep = self.num_v_heads // self.num_heads
            q = jnp.repeat(q, rep, axis=2)
            k = jnp.repeat(k, rep, axis=2)
            
        beta = nnx.sigmoid(self.b_proj(x))
        g_in = self.a_proj(x) + self.dt_bias
        g = -jnp.exp(self.A_log) * nnx.softplus(g_in)
        
        if self.config.attn_mode == 'chunk':
            o, _ = chunk_gated_delta_rule_fwd(q, k, v, g, beta)
        elif self.config.attn_mode == 'fused_recurrent':
            o, _ = fused_recurrent_gated_delta_rule_fwd(q, k, v, g, beta)
        else:
            raise ValueError(f"Unknown mode {self.config.attn_mode}")
            
        if self.config.use_gate:
            gate_out = self.g_proj(x)
            gate_out = gate_out.reshape(B, T, self.num_v_heads, self.head_v_dim)
            o = self.o_norm(o, gate=gate_out)
        else:
            o = self.o_norm(o)
            
        o = o.reshape(B, T, -1)
        o = self.o_proj(o)
        return o

# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------

class GatedDeltaNetMLP(nnx.Module):
    def __init__(self, config: GatedDeltaNetConfig, rngs: nnx.Rngs = None):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * config.hidden_ratio * 2 / 3)
            # Make multiple of 256 for efficiency
            self.intermediate_size = 256 * ((self.intermediate_size + 256 - 1) // 256)
            
        self.gate_proj = nnx.Linear(self.hidden_size, self.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(self.hidden_size, self.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(self.intermediate_size, self.hidden_size, use_bias=False, rngs=rngs)
        
    def __call__(self, x):
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))

# -----------------------------------------------------------------------------
# Block
# -----------------------------------------------------------------------------

class GatedDeltaNetBlock(nnx.Module):
    def __init__(self, config: GatedDeltaNetConfig, layer_idx: int, rngs: nnx.Rngs = None):
        self.attn_norm = nnx.RMSNorm(config.hidden_size, epsilon=config.norm_eps, rngs=rngs)
        self.attn = GatedDeltaNet(config, layer_idx=layer_idx, rngs=rngs)
        self.mlp_norm = nnx.RMSNorm(config.hidden_size, epsilon=config.norm_eps, rngs=rngs)
        self.mlp = GatedDeltaNetMLP(config, rngs=rngs)
        
    def __call__(self, x, mask=None):
        # Attention
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask=mask)
        x = residual + x
        
        # MLP
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

# -----------------------------------------------------------------------------
# Backbone Model
# -----------------------------------------------------------------------------

class GatedDeltaNetModel(nnx.Module):
    def __init__(self, config: GatedDeltaNetConfig, rngs: nnx.Rngs = None):
        self.config = config
        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.layers = nnx.List([
            GatedDeltaNetBlock(config, layer_idx=i, rngs=rngs)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = nnx.RMSNorm(config.hidden_size, epsilon=config.norm_eps, rngs=rngs)
        
    def __call__(self, input_ids):
        # input_ids: (B, T)
        x = self.embed_tokens(input_ids)
        mask = None # Add mask handling if needed later
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = self.norm(x)
        return x

# -----------------------------------------------------------------------------
# Causal LM Head
# -----------------------------------------------------------------------------

class GatedDeltaNetForCausalLM(nnx.Module):
    def __init__(self, config: GatedDeltaNetConfig, rngs: nnx.Rngs = None):
        self.config = config
        self.model = GatedDeltaNetModel(config, rngs=rngs)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)
        
    def __call__(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
