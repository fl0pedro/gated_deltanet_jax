# Gated DeltaNet JAX

JAX/Flax implementation of Gated DeltaNet, featuring:
- `chunk` and `fused_recurrent` modes (using Pallas).
- Flax NNX module structure.

## Installation

```bash
pip install -e .
```

## Usage

```python
from gated_deltanet_jax import GatedDeltaNetModel, GatedDeltaNetConfig

config = GatedDeltaNetConfig()
model = GatedDeltaNetModel(config)
```

## Benchmarks

Run the benchmark script:

```bash
uv run python benchmark_jax.py
```
