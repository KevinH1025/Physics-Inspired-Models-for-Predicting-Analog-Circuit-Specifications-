"""Benchmark scatter determinism approaches for GENConv."""
import os, sys, time
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch_scatter
from torch_sparse import SparseTensor
from torch_geometric.nn import GENConv

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = 'cuda'

# ---- Create a realistic graph (similar to opamp circuit batch) ----
N, D, E = 1340, 128, 5000  # ~10 graphs * 134 nodes, 500 edges each
edge_index = torch.stack([
    torch.randint(0, N, (E,)),
    torch.randint(0, N, (E,)),
], dim=0).to(device)
x = torch.randn(N, D, device=device)

# Sort by destination (col) for segment_csr
idx = torch.argsort(edge_index[1] * N + edge_index[0])
sorted_ei = edge_index[:, idx]

# Create SparseTensor (transposed adjacency: row=dst, col=src for source_to_target)
adj_t = SparseTensor(row=sorted_ei[1], col=sorted_ei[0], sparse_sizes=(N, N)).to(device)

conv = GENConv(D, D).to(device)

print("=== Determinism Test ===")

# Test 1: Regular edge_index — check variance across runs
print("\n[Regular edge_index]")
results = []
for i in range(10):
    torch.manual_seed(42)
    out = conv(x, sorted_ei)
    results.append(out.detach().clone())
diffs = [torch.abs(results[0] - r).max().item() for r in results[1:]]
print(f"  Max diff across 10 forward passes: {max(diffs):.2e}")
print(f"  All identical: {all(d == 0.0 for d in diffs)}")

# Test 2: SparseTensor — check variance
print("\n[SparseTensor (adj_t)]")
results_st = []
for i in range(10):
    torch.manual_seed(42)
    out = conv(x, adj_t)
    results_st.append(out.detach().clone())
diffs_st = [torch.abs(results_st[0] - r).max().item() for r in results_st[1:]]
print(f"  Max diff across 10 forward passes: {max(diffs_st):.2e}")
print(f"  All identical: {all(d == 0.0 for d in diffs_st)}")

print("\n=== Speed Benchmark (20 fwd+bwd iterations) ===")

# Warmup
for _ in range(5):
    out = conv(x, sorted_ei)
    out.sum().backward()
    conv.zero_grad()

# Benchmark: regular edge_index
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    out = conv(x, sorted_ei)
    out.sum().backward()
    conv.zero_grad()
torch.cuda.synchronize()
t_ei = (time.time() - t0) / 50
print(f"Regular edge_index: {t_ei*1000:.1f}ms/iter")

# Warmup SparseTensor
for _ in range(5):
    out = conv(x, adj_t)
    out.sum().backward()
    conv.zero_grad()

# Benchmark: SparseTensor
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    out = conv(x, adj_t)
    out.sum().backward()
    conv.zero_grad()
torch.cuda.synchronize()
t_st = (time.time() - t0) / 50
print(f"SparseTensor:       {t_st*1000:.1f}ms/iter")
print(f"Ratio: {t_st/t_ei:.2f}x")

# Benchmark: deterministic algorithms
torch.use_deterministic_algorithms(True)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    out = conv(x, sorted_ei)
    out.sum().backward()
    conv.zero_grad()
torch.cuda.synchronize()
t_det = (time.time() - t0) / 50
torch.use_deterministic_algorithms(False)
print(f"Deterministic mode: {t_det*1000:.1f}ms/iter")
print(f"Ratio: {t_det/t_ei:.2f}x")
