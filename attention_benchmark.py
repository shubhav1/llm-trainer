import torch
import time
from model import scaled_dot_product_attention
from contextlib import nullcontext

def nvtx_range(name):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return nullcontext()

def main ():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA is not available. Exiting.")
        return

    B = 8
    d_model = [16, 32, 64, 128]
    seq_len = [256, 1024, 4096, 8192, 16384]

    for d in d_model:
        for s in seq_len:
            try:
                Q = torch.randn(B, s, d, device=device, requires_grad=True)
                K = torch.randn(B, s, d, device=device, requires_grad=True)
                V = torch.randn(B, s, d, device=device, requires_grad=True)

                causal_mask = torch.tril(torch.ones(s, s, device=device, dtype=torch.bool))

                grad = torch.ones(B, s, d, device=device)
                # warmup
                for _ in range(5):
                    out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
                    torch.cuda.synchronize()

                    out.backward(grad)
                    torch.cuda.synchronize()

                    # zero grad
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                # one iteration for memory profiling
                torch.cuda.empty_cache()
                torch.cuda.memory._record_memory_history(max_entries=1000000)
                torch.cuda.synchronize()
                with nvtx_range("forward"):
                    out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
                    torch.cuda.synchronize()

                memory_before_backward = torch.cuda.memory_allocated()

                with nvtx_range("backward"):
                    out.backward(grad)
                    torch.cuda.synchronize()

                Q.grad = None
                K.grad = None
                V.grad = None
                snapshot_name = f"d_model_{d}_seqlen_{s}.pickle"
                torch.cuda.memory._dump_snapshot(snapshot_name)
                torch.cuda.memory._record_memory_history(enabled=None)

                # time 100 iterations
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                forward_times = []
                backward_times = []
                for _ in range(100):
                    # forward
                    start.record()
                    out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
                    end.record()
                    torch.cuda.synchronize()
                    peak_forward_memory = torch.cuda.max_memory_allocated()
                    
                    forward_times.append(start.elapsed_time(end))

                    # backward
                    start.record()
                    out.backward(grad)
                    end.record()
                    torch.cuda.synchronize()
                    backward_times.append(start.elapsed_time(end))

                    # zero grad
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                avg_forward_ms = sum(forward_times) / len(forward_times)

                avg_backward_ms = sum(backward_times) / len(backward_times)

                print(
                    f"d={d}, seq_len={s}: "
                    f"forward={avg_forward_ms:.3f} ms, "
                    f"backward={avg_backward_ms:.3f} ms, "
                    f"memory={memory_before_backward / 1024**3:.3f} GiB, "
                    f"memory={peak_forward_memory / 1024**3:.3f} GiB"
                )

            except torch.OutOfMemoryError:
                print(f"d={d}, seq_len={s}: CUDA OOM -- skipping")
                Q = K = V = grad = out = causal_mask = None
                torch.cuda.empty_cache()
                
                continue





if __name__ == "__main__":
    main()