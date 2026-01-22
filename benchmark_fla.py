import time
from itertools import product
import torch
import torch.nn as nn
from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel

class GatedDeltaNetClassifier(nn.Module):
    def __init__(self, vocab_size, embd_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        config = GatedDeltaNetConfig(
            hidden_size=embd_dim,
            num_heads=num_heads,
            num_hidden_layers=num_layers,
            vocab_size=vocab_size,
            attn_mode='chunk', # PyTorch FLA usually uses chunk/fused automatically or explicitly
            head_dim=embd_dim // num_heads if num_heads > 0 else embd_dim,
            intermediate_size=embd_dim * 4,
            use_gate=True,
            use_short_conv=True,
            fuse_norm=True,
            fuse_swiglu=True
        )
        self.model = GatedDeltaNetModel(config)
        self.head = nn.Linear(embd_dim, 2)

    def forward(self, x):
        # x: (B, T)
        outputs = self.model(x)
        # outputs.last_hidden_state: (B, T, D)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1) # (B, D)
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
    device='cuda'
):
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
            num_heads=num_heads
        ).to(device)
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

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y_target = torch.zeros((batch_size, 2), device=device)
    y_target[:, 0] = 1.0

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warmup Forward
    try:
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
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

    # Measure Forward
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    _ = model(x)
    torch.cuda.synchronize()
    end = time.time()

    fwd_run_time = end - start
    fwd_peak_mem = torch.cuda.max_memory_allocated()

    # Warmup Backward
    try:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y_target)
        loss.backward()
        torch.cuda.synchronize()
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

    # Measure Backward
    # Note: We measure the whole step (fwd+bwd) often, but here specifically Backward?
    # The script seems to separate Fw and Bwd.
    # We will measure backward-only time if possible, but usually we need fwd graph.
    # To measure PURE backward time, we do fwd, then sync, then time .backward()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y_target)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    end = time.time()

    bwd_run_time = end - start
    bwd_peak_mem = torch.cuda.max_memory_allocated() # Logic mismatch: max mem during backward might include fwd activations?
    # Actually `reset_peak_memory_stats` before backward should capture peak during backward.
    # But backward relies on retained graph. The "Peak" might be high due to retained graph.
    # This is fine, comparable to JAX.

    return {
        "Model": model_name,
        "Batch": batch_size,
        "SeqLen": seq_len,
        "Vocab": vocab_size,
        "Dim": embd_dim,
        "Layers": num_layers,
        "Heads": num_heads,
        "FwdCompile(s)": "N/A", # PyTorch Eager (mostly)
        "FwdRun(s)": f"{fwd_run_time:.4f}",
        "FwdMem(MB)": f"{fwd_peak_mem / 1e6:.2f}",
        "BwdCompile(s)": "N/A",
        "BwdRun(s)": f"{bwd_run_time:.4f}",
        "BwdMem(MB)": f"{bwd_peak_mem / 1e6:.2f}",
        "Status": "OK"
    }

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    header = [
        "Model", "Batch", "SeqLen", "Vocab", "Dim", "Layers", "Heads",
        "FwdCompile(s)", "FwdRun(s)", "FwdMem(MB)",
        "BwdCompile(s)", "BwdRun(s)", "BwdMem(MB)", "Status"
    ]
    print("\t".join(header))

    batch_sizes = [16, 32, 64, 128]
    seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    vocab_sizes = [128, 256]
    embed_dims = [64, 128, 256]
    layer_counts = [2, 4, 6]
    head_counts = [2, 4, 6, 8, 10, 12]

    for b, s, v, d, l, h in product(
        batch_sizes, seq_lens, vocab_sizes, embed_dims, layer_counts, head_counts
    ):
        stats = benchmark_run(
            "Torch_GatedDeltaNet", GatedDeltaNetClassifier, b, s, v, d, l, h
        )
        row = [str(stats.get(col, "")) for col in header]
        print("\t".join(row))

if __name__ == "__main__":
    main()

