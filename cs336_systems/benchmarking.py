import torch
import timeit
import statistics
import sys
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36)
}

def run_step(model, optimizer, x, y, mode="full", precision="fp32"):
    """
    Run one model step. Mode options: "forward", "forward_backward", "full". Precision options: "fp32", "bf16".
    """

    device = x.device.type

    optimizer.zero_grad()

    # mixed precision if specified
    if precision == "bf16" and device == "cuda":
        context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        context = nullcontext()

    # forward
    with nvtx.range("forward_pass"):
        with context:
            logits = model(x)

    # stop if mode = forward
    if mode == "forward":
        return logits

    # loss + backward
    with nvtx.range("backward_pass"):
        # new context for loss computation
        if precision == "bf16" and device == "cuda":
            loss_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            loss_context = nullcontext()

        with loss_context:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()

    # stop if mode = forward_backward
    if mode == "forward_backward":
        return loss

    # optimizer step
    with nvtx.range("optimizer_step"):
        optimizer.step()

    return loss


def benchmarking_script(d_model, d_ff, num_layers, num_heads, w, n, context_length=512,
                        batch_size=4, mode="full", precision="fp32", measurement="timing"):
    """
    Initialize a Transformer model and benchmark/profile it.
    mode options: "forward", "forward_backward", "full"
    precision options: "fp32", "bf16"
    measurement options: "timing", "profile", "memory"
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize model based on given hyperparameters
    vocab_size = 10000
    model = BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model,
                                num_layers=num_layers, num_heads=num_heads, d_ff=d_ff).to(device)

    # generate random batch
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    optimizer = AdamW(model.parameters())

    # warmup steps
    for _ in range(w):
        run_step(model=model,  optimizer=optimizer, x=x, y=y, mode=mode, precision=precision)
        torch.cuda.synchronize() if device == "cuda" else None

    # timing benchmark
    if measurement == "timing":
        times = []

        torch.cuda.synchronize() if device == "cuda" else None
        for step in range(n):
            start_time = timeit.default_timer()

            run_step(model=model,  optimizer=optimizer, x=x, y=y, mode=mode, precision=precision)

            torch.cuda.synchronize() if device == "cuda" else None
            times.append(timeit.default_timer() - start_time)

        return {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        }

    # Nsight profiling
    elif measurement == "profile":
        for step in range(n):
            with nvtx.range(f"step_{step}"):
                run_step(model=model,  optimizer=optimizer, x=x, y=y, mode=mode, precision=precision)
                torch.cuda.synchronize() if device == "cuda" else None

        return None

    # pytorch memory profiling
    elif measurement == "memory" and device == "cuda":

        # clear empty cache
        torch.cuda.empty_cache()

        # start recording memory history.
        torch.cuda.memory._record_memory_history(max_entries=1000000)

        # profile n steps
        for step in range(n):
            with nvtx.range(f"step_{step}"):
                run_step(model=model,  optimizer=optimizer, x=x, y=y, mode=mode, precision=precision)
                torch.cuda.synchronize()

        # save a pickle file to be loaded by pytorch's online tool
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

        # stop recording history
        torch.cuda.memory._record_memory_history(enabled=None)

        return None

    else:
        raise ValueError(f"unable to run {measurement}, or on device {device}")


def main():
    # get command line inputs
    # benchmarking.py model_name context_length warmup_steps measured_steps mode precision measurement
    # benchmarking.py small 512 5 5 full fp32 timing
    model_name = sys.argv[1]
    context_length = int(sys.argv[2])
    warmup_steps = int(sys.argv[3])
    measured_steps = int(sys.argv[4])
    mode = sys.argv[5]
    precision = sys.argv[6]
    measurement = sys.argv[7]

    # catch errors before jumping into runs
    if mode not in ("forward", "forward_backward", "full"):
        raise ValueError("mode must be forward, forward_backward, or full")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if measurement not in ("timing", "profile", "memory"):
        raise ValueError("measurement must be timing, profile, or memory")
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"unknown model: {model_name}")

    batch_size = 4
    
    print(f"Model:          {model_name}")
    print(f"Context length: {context_length}")
    print(f"Batch size:     {batch_size}")
    print(f"Mode:           {mode}")
    print(f"Precision:      {precision}")
    print(f"Measurement:    {measurement}")
    print(f"Warmup steps:   {warmup_steps}")
    print(f"Measured steps: {measured_steps}\n")

    result = benchmarking_script(**MODEL_CONFIGS[model_name], w=warmup_steps, n=measured_steps,
                                 context_length=context_length, batch_size=batch_size, mode=mode,
                                 precision=precision, measurement=measurement)

    if measurement == "timing":
        print(f"Mean: {result['mean']:.6f}s")
        print(f"Std:  {result['std']:.6f}s")

    elif measurement == "profile":
        print("Profiling run complete.")

    elif measurement == "memory":
        print(f"Memory snapshot written to memory_snapshot.pickle")

if __name__ == "__main__":
    main()

"""
Timing analysis/results:

5 warmup steps, 10 timed steps
Model: small
Forward only:         Mean (0.0169s), Std (0.0002s)
Forward and backward: Mean (0.0487s), Std (0.0001s)
Full training step:   Mean (0.0563s), Std (0.0001s)

Model: medium
Forward only:         Mean (0.0477s), Std (0.0004s)
Forward and backward: Mean (0.1393s), Std (0.0001s)
Full training step:   Mean (0.1608s), Std (0.0008s)

Model: large
Forward only:         Mean (0.1063s), Std (0.0001s)
Forward and backward: Mean (0.3133s), Std (0.0002s)
Full training step:   Mean (0.3486s), Std (0.0016s)

Model: xl
Forward only:         Mean (0.2962s), Std (0.0002s)
Forward and backward: Mean (0.8652s), Std (0.0005s)
Full training step:   Mean (0.9447s), Std (0.0002s)

Model: 10B
unable to run due to memory constraints on the GPU

Analysis: Forward + backward pass takes 2.5-3x longer than forward pass alone, indicating that 
the backward pass takes 50-100% longer than forward. The full steps adds a bit of time (~10-15%).
In the medium and large models, standard deviation is comparitively high for full steps, though
this doesn't hold consistent for xl and small. This isn't enough information to differentiate if 
this is a real finding or just noise. Besides that, std dev is generally low across models/steps.

-------------------------------------------------------------------------------------------------
0 warmup steps, 10 timed steps
Model: small
Forward only:         Mean (0.0385s), Std (0.0698s)
Forward and backward: Mean (0.0680s), Std (0.0631s)
Full training step:   Mean (0.0585s), Std (0.0085s)

Model: medium
Forward only:         Mean (0.0475s), Std (0.0010s)
Forward and backward: Mean (0.1389s), Std (0.0001s)
Full training step:   Mean (0.1558s), Std (0.0019s)

Model: large
Forward only:         Mean (0.1094s), Std (0.0094s)
Forward and backward: Mean (0.3130s), Std (0.0002s)
Full training step:   Mean (0.3424s), Std (0.0004s)

Model: xl
Forward only:         Mean (0.2951s), Std (0.0005s)
Forward and backward: Mean (0.8614s), Std (0.0007s)
Full training step:   Mean (0.9421s), Std (0.0045s)

Model: 10B
unable to run due to memory constraints on the GPU

Analysis: Removing the warmup impacts the small model the most, perhaps because the initial 
iterations impact the mean/std dev more than in larger models. In the small model, forward and 
forward + backward are significantly higher without warmup. The increase is less apparent in small 
model's full step and in other models. Std dev is higher for most of the steps across all models, 
though the difference is highest in the small model.

-------------------------------------------------------------------------------------------------
2 warmup steps, 10 timed steps
Model: small
Forward only:         Mean (0.0164s), Std (0.0000s)
Forward and backward: Mean (0.0481s), Std (0.0002s)
Full training step:   Mean (0.0555s), Std (0.0002s)

Model: medium
Forward only:         Mean (0.0471s), Std (0.0003s)
Forward and backward: Mean (0.1387s), Std (0.0001s)
Full training step:   Mean (0.1558s), Std (0.0017s)

Model: large
Forward only:         Mean (0.1062s), Std (0.0003s)
Forward and backward: Mean (0.3128s), Std (0.0002s)
Full training step:   Mean (0.3424s), Std (0.0002s)

Model: xl
Forward only:         Mean (0.2934s), Std (0.0005s)
Forward and backward: Mean (0.8610s), Std (0.0004s)
Full training step:   Mean (0.9406s), Std (0.0005s)

Model: 10B
unable to run due to memory constraints on the GPU

Analysis: Adding 2 warmup iterations gets the means across all steps/models to around the same as
when there are 5 warmup iterations. Std dev is also similar, though there is a bit of noise where
some scenarios have higher and some have lower std dev than the 5 warmup iteration scenario.

"""


"""
NSight profiling analysis/results:

for 512 context len, small size, over 5 observed profiled steps, ran on A100:

These are insights based on the profiling results viewed separately on the NVIDIA Nsight Systems GUI.

Forward pass is taking ~44 ms for CUDA HW NVTX projection, but on CPU (python3) it is taking ~24 ms. Both start
at the same time (same as when the step itself starts), but the GPU is still computing by the time CPU moves onto
the next step.

During forward pass, the kernel that takes up most of the time is ampere_sgemm_128x32_tn, taking 32% of kernel 
time. It occurs 24 times. This is not the same as the kernel that takes the most time in forward + backward, 
that one is different size (ampere_sgemm_64x32_sliced1x4_nt).

Asside from matmuls, the kernels that take up non trivial time in forward pass are variations of elementwise 
and vectorized elementwise kernels.

The ampere_sgemm (matmul) kernels tak up 78.4% of CUDA HW time in the forward pass, however they take up only
69.3% of CUDA HW time in the entire step (forward + backward + optimizer step). Other kernels take up time across
the entire step (not just elementwise, but also others like reduce, scatter gather, SoftMax specific stuff, etc.)
"""