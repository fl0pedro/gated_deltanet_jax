import time
from itertools import product
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx
import optax
from configuration import GatedDeltaNetConfig
from model import GatedDeltaNetModel
from memory_monitor import PeakMemoryMonitor

class GatedDeltaNetClassifier(nnx.Module):
    def __init__(self, vocab_size, embd_dim, hidden_dim, num_layers, num_heads, key):
        config = GatedDeltaNetConfig(
            hidden_size=embd_dim,
            num_heads=num_heads,
            num_hidden_layers=num_layers,
            vocab_size=vocab_size,
            attn_mode='fused_recurrent', # Focusing on fused recurrent for speed/memory
            head_dim=embd_dim // num_heads if num_heads > 0 else embd_dim,
            intermediate_size=embd_dim * 4, # Standard expansion
            use_gate=True,
            use_short_conv=True
        )
        self.model = GatedDeltaNetModel(config, rngs=nnx.Rngs(key))
        self.head = nnx.Linear(embd_dim, 2, rngs=nnx.Rngs(key))

    def __call__(self, x):
        # x is (B, T)
        outputs = self.model(x)
        # pooled = jnp.mean(outputs, axis=1) # Mean pooling
        # Use last token for classification to be more realistic for RNNs?
        # Or mean. Let's use mean as in the template (implied by "pred - y" if y is (B, 2))
        # The template had y_target as (B, 2) and used means.
        # Actually template has y_target (B, 2).
        # We need (B, 2).
        pooled = jnp.mean(outputs, axis=1)
        return self.head(pooled)

def benchmark_run(
    model_name,
    model_class,
    batch_size,
    seq_len,
    vocab_size,
    embd_dim,
    num_layers,
    num_heads,
    key,
):
    model_key, data_key = jr.split(key)

    # Check validity
    if num_heads > 0 and embd_dim % num_heads != 0:
        return {
            "Model": model_name,
            "Batch": batch_size,
            "SeqLen": seq_len,
            "Vocab": vocab_size,
            "Dim": embd_dim,
            "Layers": num_layers,
            "Heads": num_heads,
            "Status": "InvalidConfig" 
        }

    try:
        model = model_class(
            vocab_size=vocab_size,
            embd_dim=embd_dim,
            hidden_dim=embd_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            key=model_key,
        )
    except Exception as e:
         return {
            "Model": model_name,
            "Batch": batch_size,
            "SeqLen": seq_len,
            "Vocab": vocab_size,
            "Dim": embd_dim,
            "Layers": num_layers,
            "Heads": num_heads,
            "Status": f"InitError: {e}" 
        }

    x = jr.randint(data_key, (batch_size, seq_len), 0, vocab_size)
    y_target = jnp.zeros((batch_size, 2))
    y_target = y_target.at[:, 0].set(1.0)

    # Compile Forward
    # NNX handles state. We need to be careful with JIT and NNX.
    # nnx.jit allows jitting the call.

    @nnx.jit
    def forward(m, x):
        return m(x)

    @nnx.jit
    def backward(m, x, y):
        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)
        
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(m, x, y)
        return grads

    # Warmup / Compile Forward
    start = time.time()
    try:
        out = forward(model, x)
        jax.block_until_ready(out)
    except Exception as e:
         return {
            "Model": model_name,
            "Batch": batch_size,
            "SeqLen": seq_len,
            "Vocab": vocab_size,
            "Dim": embd_dim,
            "Layers": num_layers,
            "Heads": num_heads,
            "Status": f"FwdError: {e}" 
        }
    end = time.time()
    fwd_compile_time = end - start

    # Run Forward
    start = time.time()
    with PeakMemoryMonitor(interval=0.01) as mem:
        out = forward(model, x)
        jax.block_until_ready(out)
    end = time.time()
    fwd_run_time = end - start
    fwd_peak_mem = mem.peak

    # Warmup / Compile Backward
    start = time.time()
    try:
        grads = backward(model, x, y_target)
        jax.tree_util.tree_map(lambda l: l.block_until_ready(), grads)
    except Exception as e:
         return {
            "Model": model_name,
            "Batch": batch_size,
            "SeqLen": seq_len,
            "Vocab": vocab_size,
            "Dim": embd_dim,
            "Layers": num_layers,
            "Heads": num_heads,
            "Status": f"BwdError: {e}" 
        }
    end = time.time()
    bwd_compile_time = end - start

    # Run Backward
    start = time.time()
    with PeakMemoryMonitor(interval=0.01) as mem:
        grads = backward(model, x, y_target)
        jax.tree_util.tree_map(lambda l: l.block_until_ready(), grads)
    end = time.time()
    bwd_run_time = end - start
    bwd_peak_mem = mem.peak

    return {
        "Model": model_name,
        "Batch": batch_size,
        "SeqLen": seq_len,
        "Vocab": vocab_size,
        "Dim": embd_dim,
        "Layers": num_layers,
        "Heads": num_heads,
        "FwdCompile(s)": f"{fwd_compile_time:.4f}",
        "FwdRun(s)": f"{fwd_run_time:.4f}",
        "FwdMem(MB)": f"{fwd_peak_mem / 1e6:.2f}",
        "BwdCompile(s)": f"{bwd_compile_time:.4f}",
        "BwdRun(s)": f"{bwd_run_time:.4f}",
        "BwdMem(MB)": f"{bwd_peak_mem / 1e6:.2f}",
        "Status": "OK"
    }


def main():
    header = [
        "Model", "Batch", "SeqLen", "Vocab", "Dim", "Layers", "Heads",
        "FwdCompile(s)", "FwdRun(s)", "FwdMem(MB)",
        "BwdCompile(s)", "BwdRun(s)", "BwdMem(MB)", "Status"
    ]
    print("\t".join(header))

    key = jr.PRNGKey(42)

    # Reduced set for quick validation, uncomment full range for final run if requested
    # User asked for:
    # batch_sizes = [16, 32, 64, 128]
    # seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    # vocab_sizes = [128, 256]
    # embed_dims = [64, 128, 256]
    # layer_counts = [2, 4, 6]
    # head_counts = [2, 4, 6, 8, 10, 12]
    
    # This is huge. I'll define them but maybe just run a few if I can't run all.
    # The user explicitly asked "Benchmark the following".
    # I should try to run them, but 2000+ runs will take hours.
    # I will limit to a smaller subset for the 'small benchmark' requested initially, 
    # or just one combination of each to show it works, then maybe the user can run the full thing.
    # "make a small benchmark ... Benchmark the following"
    # I'll implement the loops but maybe break after 1 step to show it works?
    # No, I'll run a subset: Batch=16/32, Seq=2048/4096, Dim=64, Layers=2, Heads=2/4
    # Just to verify.
    
    # batch_sizes = [16, 32]
    # seq_lens = [2048, 4096]
    # vocab_sizes = [128]
    # embed_dims = [64]
    # layer_counts = [2]
    # head_counts = [2, 4]

    # Full set
    batch_sizes = [16, 32, 64, 128]
    seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    vocab_sizes = [128, 256]
    embed_dims = [64, 128, 256]
    layer_counts = [2, 4, 6]
    head_counts = [2, 4, 6, 8, 10, 12]

    for b, s, v, d, l, h in product(
        batch_sizes, seq_lens, vocab_sizes, embed_dims, layer_counts, head_counts
    ):
        key, subkey = jr.split(key)
        stats = benchmark_run(
            "JAX_GatedDeltaNet", GatedDeltaNetClassifier, b, s, v, d, l, h, subkey
        )
        row = [str(stats.get(col, "")) for col in header]
        print("\t".join(row))

if __name__ == "__main__":
    main()
