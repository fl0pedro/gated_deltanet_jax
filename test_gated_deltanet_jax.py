
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from configuration import GatedDeltaNetConfig
from model import GatedDeltaNet, GatedDeltaNetForCausalLM

def test_layer_forward():
    print("Testing GatedDeltaNet Layer [Chunk]...")
    config = GatedDeltaNetConfig(
        hidden_size=128,
        head_dim=32,
        num_heads=4,
        attn_mode='chunk'
    )
    rngs = nnx.Rngs(0)
    model = GatedDeltaNet(config, rngs=rngs)
    
    x = jnp.ones((1, 64, 128))
    y = model(x)
    print("Output shape:", y.shape)
    assert y.shape == (1, 64, 128)
    print("Layer Test Passed!")

def test_causal_lm_forward():
    print("Testing GatedDeltaNetForCausalLM [Pallas]...")
    config = GatedDeltaNetConfig(
        hidden_size=64,
        head_dim=16,
        num_heads=4,
        num_hidden_layers=2,
        vocab_size=100,
        attn_mode='fused_recurrent'
    )
    rngs = nnx.Rngs(1)
    model = GatedDeltaNetForCausalLM(config, rngs=rngs)
    
    input_ids = jnp.zeros((1, 32), dtype=jnp.int32)
    logits = model(input_ids)
    print("Logits shape:", logits.shape)
    assert logits.shape == (1, 32, 100)
    print("CausalLM Test Passed!")

if __name__ == "__main__":
    test_layer_forward()
    test_causal_lm_forward()
